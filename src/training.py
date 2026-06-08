from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


@dataclass
class SupervisedDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _apply_qwen_chat_template(tokenizer, messages, *, add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    # Qwen3 accepts enable_thinking. Other tokenizers may not.
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def tokenize_sft_dataset(dataset: Dataset, tokenizer, max_seq_length: int = 1024) -> Dataset:
    def convert(example):
        prompt = str(example["prompt"])
        response = str(example["response"])

        prompt_messages = [{"role": "user", "content": prompt}]
        full_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        prompt_text = _apply_qwen_chat_template(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
        full_text = _apply_qwen_chat_template(
            tokenizer,
            full_messages,
            add_generation_prompt=False,
        )

        if tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
            full_text += tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = full["input_ids"]
        labels = list(input_ids)

        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    return dataset.map(convert, remove_columns=dataset.column_names)


def load_model_for_lora(
    model_name: str,
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",
):
    dtype = torch.bfloat16 if bnb_4bit_compute_dtype == "bfloat16" else torch.float16
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    return model


def build_lora_config(
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
        or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def train_lora(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    model_name: str,
    output_dir: str,
    max_seq_length: int = 1024,
    load_in_4bit: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 10,
    logging_steps: int = 10,
    save_steps: int = 200,
    seed: int = 42,
):
    tokenizer = load_tokenizer(model_name)
    tokenized_train = tokenize_sft_dataset(train_dataset, tokenizer, max_seq_length=max_seq_length)
    tokenized_val = tokenize_sft_dataset(val_dataset, tokenizer, max_seq_length=max_seq_length) if val_dataset else None

    model = load_model_for_lora(model_name=model_name, load_in_4bit=load_in_4bit)
    lora_config = build_lora_config(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="steps" if tokenized_val is not None else "no",
        eval_steps=save_steps if tokenized_val is not None else None,
        optim="paged_adamw_32bit" if load_in_4bit else "adamw_torch",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=SupervisedDataCollator(tokenizer),
    )
    trainer.train()

    final_dir = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    return final_dir
