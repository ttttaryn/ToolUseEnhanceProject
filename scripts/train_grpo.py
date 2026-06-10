from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from trl_grpo_tooluse.rewards import full_rewards, selection_only_rewards


def main() -> None:
    args = parse_args()
    dataset = load_dataset("json", data_files=str(args.dataset), split="train")
    if args.max_train_samples:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path or args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args)
    peft_config = None
    if args.use_lora and args.base_model_path is None:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else None,
        )

    rewards = (
        selection_only_rewards()
        if args.reward_preset == "selection_only"
        else full_rewards(include_hallucination_penalty=not args.disable_hallucination_penalty)
    )

    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        beta=args.beta,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        bf16=args.bf16,
        fp16=args.fp16,
        use_vllm=args.use_vllm,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=rewards,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


def load_model(args: argparse.Namespace):
    dtype = "auto"
    quantization_config = None
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("--load-in-4bit requires bitsandbytes-compatible transformers install") from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if args.base_model_path:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, args.model_path, is_trainable=True)
        return model
    return args.model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BFCL tool-use GRPO with TRL.")
    parser.add_argument("--model-path", required=True, help="SFT model path or SFT LoRA adapter path.")
    parser.add_argument("--base-model-path", default=None, help="Base model path when --model-path is a LoRA adapter.")
    parser.add_argument("--dataset", required=True, type=Path, help="Prepared JSONL dataset.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reward-preset", choices=["full", "selection_only"], default="full")
    parser.add_argument("--disable-hallucination-penalty", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable optional QLoRA-style 4-bit loading.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    return parser.parse_args()


if __name__ == "__main__":
    main()
