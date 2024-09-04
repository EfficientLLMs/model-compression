from transformers import GPTNeoXForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
import torch
import torch.nn as nn
from mora_fine_tune import evaluate_model
from accelerate import Accelerator
import pandas as pd
import argparse


def truncate_mora_weights(old_module, new_rank):

    new_module = nn.Linear(new_rank, new_rank, bias=old_module.bias is not None)

    with torch.no_grad():
        new_module.weight.copy_(old_module.weight[:new_rank, :new_rank])

        if old_module.bias is not None:
            new_module.bias.copy_(old_module.bias[:new_rank])

    return new_module


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--large_model", type=str, default="410m")
    parser.add_argument("--small_model", type=str, default="70m")
    parser.add_argument("--large_adapter", type=str, default="./weight/pythia_410m_r=8_0.0001_fixed")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--mora_type", type=int, default=6)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_dataloader", type=str, default="./data/eval_dataloader.pt")
    parser.add_argument("--new_rhat", type=int, default=128)
    parser.add_argument("--output_csv", type=str, default="./eval/evaluate_410m_70m.csv")
    
    args = parser.parse_args()
    
    accelerator = Accelerator()
    
    eval_dataloader = torch.load(args.eval_dataloader)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-" + args.large_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    large_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-" + args.large_model)
    large_model.load_adapter(args.large_adapter)

    mora_config = LoraConfig(
        # enable MoRA
        use_mora=True,
        # type 1 (Sharing) for large lora ranks, Eq. 6 in paper
        # type 6 (RoPE based) for small lora ranks, Eq. 9 in paper
        mora_type=args.mora_type,
        # lora rank here, we will calculate corresponding $\hat{r}$ in MoRA
        r=args.rank,
        # MoRA does not use lora_alpha
        # lora_alpha=lora_alpha,
        target_modules=["query_key_value"],
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM"
    )

    small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-" + args.small_model)
    small_model = get_peft_model(small_model, mora_config)

    eval_results = pd.DataFrame()

    # Prepare for accelerator
    eval_dataloader, large_model, small_model, tokenizer = accelerator.prepare(
        eval_dataloader, large_model, small_model, tokenizer
    )

    # Evaluate the large model
    eval_loss, eval_rouge_scores = evaluate_model(large_model, eval_dataloader, accelerator, tokenizer)

    eval_results = pd.concat(
        [
            eval_results,
            pd.DataFrame(
                {
                    "model": "fine_tuned_" + args.large_model,
                    "rank": args.rank,
                    "eval_loss": eval_loss,
                    **eval_rouge_scores,
                }
            ),
        ]
    )

    # Evaluate the small model
    eval_loss, eval_rouge_scores = evaluate_model(small_model, eval_dataloader, accelerator, tokenizer)

    eval_results = pd.concat(
        [
            eval_results,
            pd.DataFrame(
                {
                    "model": "raw_" + args.small_model,
                    "rank": args.rank,
                    "eval_loss": eval_loss,
                    **eval_rouge_scores,
                }
            ),
        ]
    )


    # Truncate the weights of the large model to the new rank

    for name, module in large_model.named_modules():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            new_module = truncate_mora_weights(module.default, args.new_rhat)
            parts = name.split('.')
            parent_module = small_model
            for part in parts[:-1]:

                if hasattr(parent_module, part):
                    parent_module = getattr(parent_module, part)

            # Move the new module to the correct device
            new_module = new_module.to(parent_module.device)
            
            setattr(parent_module, parts[-1], nn.ModuleDict({"default": new_module}))

    # Evaluate the large model with truncated weights
    eval_loss, eval_rouge_scores = evaluate_model(large_model, eval_dataloader, accelerator, tokenizer)

    eval_results = pd.concat(
        [
            eval_results,
            pd.DataFrame(
                {
                    "model": "truncated_" + args.small_model + "_from_" + args.large_model,
                    "rank": args.rank,
                    "eval_loss": eval_loss,
                    **eval_rouge_scores,
                }
            ),
        ]
    )

    # Save the evaluation results
    eval_results.to_csv(args.output_csv, index=False)