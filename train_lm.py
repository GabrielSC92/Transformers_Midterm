#!/usr/bin/env python3
"""
Fine-tune a small causal LM on engine_data_lm.jsonl (FEN -> move) with LoRA.
Run convert_engine_data_for_lm.py first to create engine_data_lm.jsonl.

Usage:
  python convert_engine_data_for_lm.py --out engine_data_lm.jsonl
  python train_lm.py --data engine_data_lm.jsonl --output_dir chess_lm_lora --epochs 1

Requires: pip install transformers datasets peft trl accelerate bitsandbytes
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="engine_data_lm.jsonl", help="JSONL with prompt, completion")
    p.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-1.7B-Instruct", help="Base causal LM (different from Llama/Ministral)")
    p.add_argument("--output_dir", default="chess_lm_lora", help="Save dir for adapter and tokenizer")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--limit", type=int, default=None, help="Use only first N rows (for testing)")
    args = p.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"Data file not found: {args.data}. Run convert_engine_data_for_lm.py first.")

    # Load dataset: expect {"prompt": "FEN: ...\nMove:\n", "completion": "e2e4"}
    dataset = load_dataset("json", data_files=args.data, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Build full text so completion starts after "Move:\n" (for completion_only_loss)
    def add_text(example):
        example["text"] = example["prompt"] + example["completion"]
        return example

    dataset = dataset.map(add_text, num_proc=1)
    response_template = "Move:\n"  # loss only on tokens after this

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16"),
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=2000,
        completion_only_loss=True,
    )

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
