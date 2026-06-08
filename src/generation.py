from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import PromptBuilder
from .retrieval import RetrievalStore


class LMGenerator:
    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        max_new_tokens: int = 32,
        supports_thinking: bool | None = None,
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.supports_thinking = supports_thinking if supports_thinking is not None else "Qwen3" in model_name

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def format_prompt(self, prompt: str, thinking: bool = False) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.supports_thinking:
            kwargs["enable_thinking"] = thinking
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    @staticmethod
    def clean_answer(text: str) -> str:
        text = text.strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        text = text.split("\n")[0].strip()
        for prefix in ["Answer:", "The answer is:", "The answer is"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        return text.strip(" .")

    def generate_answers_batch(self, prompts: List[str], thinking: bool = False) -> List[Tuple[str, str]]:
        texts = [self.format_prompt(prompt, thinking=thinking) for prompt in prompts]
        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = model_inputs["input_ids"].shape[1]
        decoded = self.tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)
        return [(self.clean_answer(text), "") for text in decoded]


class ExperimentRunner:
    def __init__(self, prompt_builder: PromptBuilder, retrieval_store: RetrievalStore, top_k: int = 3):
        self.prompt_builder = prompt_builder
        self.retrieval_store = retrieval_store
        self.top_k = top_k

    def build_prompt_from_row(self, row, setting: str) -> Tuple[str, List[int]]:
        if setting == "baseline":
            return self.prompt_builder.build_baseline_prompt(row.query), []
        if setting == "rag":
            indices = self.retrieval_store.get_rag_indices(row)
        elif setting == "oracle":
            indices = self.retrieval_store.get_oracle_indices(row)
        else:
            raise ValueError(f"Unknown setting: {setting}")
        chunks = [row.candidate_chunks[int(i)] for i in indices]
        return self.prompt_builder.build_rag_prompt(row.query, chunks), indices

    def retrieve_answers(
        self,
        df: pd.DataFrame,
        model: LMGenerator,
        setting: str,
        batch_size: int = 8,
        thinking: bool = False,
    ) -> List[Dict]:
        df = df.to_pandas() if not isinstance(df, pd.DataFrame) else df
        prompts, metadata = [], []
        for row in df.itertuples():
            prompt, indices = self.build_prompt_from_row(row, setting)
            prompts.append(prompt)
            metadata.append({
                "query_id": row.query_id,
                "retrieved_chunks": [int(i) for i in indices],
                "augmented_prompt": prompt,
                "query": row.query,
                "short_answer": getattr(row, "short_answer", None),
                "answer_pos": getattr(row, "answer_pos", None),
            })

        results = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        for start in tqdm(range(0, len(prompts), batch_size), total=total_batches, desc=f"{setting} generation"):
            batch_prompts = prompts[start : start + batch_size]
            batch_meta = metadata[start : start + batch_size]
            batch_answers = model.generate_answers_batch(batch_prompts, thinking=thinking)
            for meta, (answer, _) in zip(batch_meta, batch_answers):
                results.append({**meta, "generated_answer": answer})
        return results
