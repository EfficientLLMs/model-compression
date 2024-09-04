import random
import os
import torch
import argparse
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import evaluate
import wandb
from accelerate import Accelerator


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dataloader(dataset_name):

    if dataset_name == "alpaca_instruction_gpt4":
        
        train_dataloader = torch.load("./data/train_dataloader.pt")
        eval_dataloader = torch.load("./data/eval_dataloader.pt")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataloader, eval_dataloader

def evaluate_model(model, test_dl, accelerator, tokenizer):
    model.eval()
    total_loss = 0
    rouge = evaluate.load('rouge')

    if accelerator.is_main_process:
        test_iter = tqdm(test_dl, desc="Evaluating")
    else:
        test_iter = test_dl

    for batch in test_iter:
        labels = batch["labels"].detach().cpu().numpy()

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            generated_ids = torch.argmax(outputs.logits, dim=-1)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            rouge.add_batch(predictions=generated_texts, references=reference_texts)
    
    avg_eval_loss = total_loss / len(test_dl)
    final_rouge_scores = rouge.compute()

    if accelerator.is_main_process:
        print(f"Average Evaluation Loss: {avg_eval_loss}")
        print("ROUGE Scores:", final_rouge_scores)

    return avg_eval_loss, final_rouge_scores

def train_model(model, optimizer, scheduler, learning_rate, train_dl, test_dl, epochs, accelerator, tokenizer, output):
    model.train()
    if accelerator.is_main_process:
        print("Start training...")

    for epoch in tqdm(range(epochs)):
        if accelerator.is_main_process:
            train_iter = tqdm(train_dl, desc=f"Epoch {epoch + 1}")
        else:
            train_iter = train_dl
            
        total_loss = 0
        for batch in train_iter:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = total_loss / len(train_dl)
        if accelerator.is_main_process:
            print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss}")

        eval_loss, eval_rouge_scores = evaluate_model(model, test_dl, accelerator, tokenizer)
        if accelerator.is_main_process:

            lr = scheduler.get_last_lr()[0] if scheduler is not None else learning_rate
            wandb.log({
                "epoch": epoch, 
                "learning_rate": lr,
                "training_loss": avg_train_loss,
                "evaluation_loss": eval_loss, 
                **eval_rouge_scores
            })
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )

    if accelerator.is_main_process:
        print("Training finished...")


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='1.4b')
    parser.add_argument("--seed", type=int, default=1006)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dataset_name", type=str, default="alpaca_instruction_gpt4")
    args = parser.parse_args()

    args.base_model = f"EleutherAI/pythia-{args.name}"

    if args.scheduler:
        args.wandb_name = f"{args.name}_r={args.rank}_{args.lr}_schedule"
        args.output = f"./weight/pythia_{args.name}_r={args.rank}_{args.lr}_schedule/"
    else:
        args.wandb_name = f"{args.name}_r={args.rank}_{args.lr}_fixed"
        args.output = f"./weight/pythia_{args.name}_r={args.rank}_{args.lr}_fixed/"
        

    # accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"device: {device}")

    # seed
    seed_everything(args.seed)

    # dataloader
    train_dataloader, eval_dataloader = get_dataloader(args.dataset_name)

    # wandb
    if accelerator.is_main_process:
        wandb.init(
            name=args.wandb_name,
            # project="dump",
            project="mora-instruction-finetune", 
            entity="vibhamasti"
        )
        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": train_dataloader.batch_size,
            "seed": args.seed
        }

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,  # standard model; the same tokenizer is used for all models
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    
    # larger base model
    base_model = GPTNeoXForCausalLM.from_pretrained(
        args.base_model,
        device_map=device,
        use_cache=False,
    )

    # config_lora = LoraConfig(
    #     r=64, 
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     target_modules=["query_key_value"],
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )

    config = LoraConfig(
        # enable MoRA
        use_mora=True,
        # type 1 (Sharing) for large lora ranks, Eq. 6 in paper
        # type 6 (RoPE based) for small lora ranks, Eq. 9 in paper
        mora_type=6,
        # lora rank here, we will calculate corresponding $\hat{r}$ in MoRA
        r=args.rank,
        # MoRA does not use lora_alpha
        # lora_alpha=lora_alpha,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        # **kwargs,
    )

    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(base_model, config)

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

        # prepare for accelerator
        model, optimizer, scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, eval_dataloader
        )

    else:

        # prepare for accelerator
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )
        
        scheduler = None

    # training and save
    train_model(model, optimizer, scheduler, args.lr, train_dataloader, eval_dataloader, args.epochs, accelerator, tokenizer, args.output)
    wandb.finish()