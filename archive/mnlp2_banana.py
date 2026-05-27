

import torch
import gc
import os
import json
import numpy as np
import pandas as pd
import random
import requests

from datasets import load_dataset, Dataset, load_from_disk

from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from typing import Dict, List, Tuple, Callable, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from tqdm.auto import tqdm

import re
import string

from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('punkt_tab')

from typing_extensions import override

torch.cuda.empty_cache()
gc.collect()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

ds_url = "sapienzanlp-course-materials/hw-mnlp-2026"
output_dir = "mnlp/hw_2"
blind_ranking_path = "https://raw.githubusercontent.com/ereznaamansapienza/mnlp/refs/heads/master/doubleN-blind-e5-mnrl-reranker-minilm-ft-full.jsonl"
test_ranking_path = "https://raw.githubusercontent.com/ereznaamansapienza/mnlp/refs/heads/master/doubleN-test-e5-mnrl-reranker-minilm-ft-full.jsonl"

SEED = 42

def set_seed(seed: int = 42):
  random.seed(seed)
  np.random.seed(seed)

  torch.manual_seed(seed)

  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

set_seed(SEED)

"""## DataModule"""

class DataModule:
    def __init__(self, dev_size: float = 0.2, seed: int = 42):
        self.ds = None
        self.dev_size = dev_size
        self.seed = seed

    # Loading
    def load(self, url: str):
        self.ds = load_dataset(url)
        return self.ds

    def to_dataframe(self, split_name: str) -> pd.DataFrame:
        return pd.DataFrame(self.ds[split_name])

    # Splits
    def build_train_val_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df_full = self.to_dataframe("train")
        train_df, val_df = train_test_split(
            train_df_full,
            test_size=self.dev_size,
            random_state=self.seed,
            shuffle=True,
        )
        return train_df, val_df

    def build_cv_splits(
        self,
        n_folds: int = 5,
        split_name: str = "train",
        stratify_col: str = "wikipedia_title",
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        df = self.to_dataframe(split_name).reset_index(drop=True)

        if stratify_col is not None:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            split_iter = kf.split(df, df[stratify_col])
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            split_iter = kf.split(df)

        folds = []
        for train_idx, val_idx in split_iter:
            folds.append((
                df.iloc[train_idx].reset_index(drop=True),
                df.iloc[val_idx].reset_index(drop=True),
            ))
        return folds

    # Dataset builders
    def build_full_train_dataset(
        self,
        dataset_factory: Callable,
        **factory_kwargs,
    ) -> Tuple[pd.DataFrame, Dataset]:
        """Builds a dataset from the entire training split with no val holdout."""
        train_df = self.to_dataframe("train")
        dataset = dataset_factory(train_df, **factory_kwargs)
        return train_df, dataset

    def build_train_val_datasets(
        self,
        dataset_factory: Callable,
        **factory_kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dataset, Dataset]:
        """Splits train into train/val and builds datasets for both."""
        train_df, val_df = self.build_train_val_split()
        train_dataset = dataset_factory(train_df, **factory_kwargs)
        val_dataset = dataset_factory(val_df,   **factory_kwargs)
        return train_df, val_df, train_dataset, val_dataset

"""## LM Generators"""

class LMGenerator():
  def __init__(self, model_name: str, max_new_tokens: int = 64, supports_thinking=False):
      self.model_name = model_name
      self.model = self.load_model()
      self.max_new_tokens = max_new_tokens
      self.supports_thinking = supports_thinking
      self.tokenizer = self.load_tokenizer()

      if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token

      # For decoder-only batched generation
      self.tokenizer.padding_side = "left"

  def load_model(self):
      return AutoModelForCausalLM.from_pretrained(
          self.model_name,
          torch_dtype="auto",
          device_map="auto"
      )

  def load_tokenizer(self):
      return AutoTokenizer.from_pretrained(self.model_name)

  def build_model_input(self, prompt: str, thinking: bool = False) -> str:
      messages = [
            {"role": "user", "content": prompt}
        ]

      kwargs = {
          "tokenize": False,
          "add_generation_prompt": True,
      }

      if self.supports_thinking:
          kwargs["enable_thinking"] = thinking

      return self.tokenizer.apply_chat_template(
          messages,
          **kwargs,
      )

  def clean_answer(self, text: str) -> str:
        text = text.strip()

        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        text = text.split("\n")[0].strip()

        for prefix in ["Answer:", "The answer is:", "The answer is"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        return text.strip(" .")

  def generate_answers_batch(
      self,
      prompts: List[str],
      thinking: bool = False,
    ) -> List[Tuple[str, str]]:
    texts = [
        self.build_model_input(prompt, thinking=thinking)
        for prompt in prompts
    ]

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

    input_length = model_inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, input_length:]

    decoded_outputs = self.tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )

    results = []
    for text in decoded_outputs:
        answer = self.clean_answer(text)
        thinking_content = ""
        results.append((answer, thinking_content))

    return results

  def generate_answer(self, prompt: str, thinking: bool = False):
      return self.generate_answers_batch([prompt], thinking=thinking)[0]

class QALLM(LMGenerator):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 64,
    ):
        super().__init__(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            supports_thinking=True,
        )

class SmolLM(LMGenerator):
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        max_new_tokens: int = 64,
    ):
        super().__init__(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            supports_thinking=False,
        )

"""## RetrievalStore"""

class RetrievalStore:
    def __init__(self, top_k: int = 3):
      self.top_k = top_k

    def load_rankings_jsonl(self, path: str) -> Dict[str, List[int]]:
        rankings = {}

        # URL case
        if path.startswith("http://") or path.startswith("https://"):
            response = requests.get(path)
            response.raise_for_status()

            lines = response.text.splitlines()

        # Local file case
        else:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        for line_no, line in enumerate(lines, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

                query_id, ranking = next(iter(obj.items()))

                rankings[str(query_id)] = [int(i) for i in ranking]

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_no}: {e}")

        return rankings

    def attach_rankings_from_jsonl(self, df: pd.DataFrame, path: str) -> pd.DataFrame:
        df = df.copy()
        rankings = self.load_rankings_jsonl(path)

        # Series.map accepts a dict-like mapping and maps values by key
        df["ranking"] = df["query_id"].astype(str).map(rankings)

        missing = df[df["ranking"].isna()]["query_id"].tolist()

        if missing:
            raise ValueError(
                f"Missing rankings for {len(missing)} query_ids. "
                f"First missing examples: {missing[:5]}"
            )

        df["retrieved_chunks"] = df["ranking"].apply(
            lambda ranking: list(ranking[:self.top_k])
        )

        self._validate_indices(df)

        return df

    def _validate_indices(self, df: pd.DataFrame) -> None:
        for row in df.itertuples():
            n_candidates = len(row.candidate_chunks)

            for idx in row.retrieved_chunks:
                if idx < 0 or idx >= n_candidates:
                    raise ValueError(
                        f"Invalid chunk index {idx} for query_id={row.query_id}. "
                        f"n_candidates={n_candidates}"
                    )

    def get_rag_indices(self, row) -> List[int]:
      return list(row.retrieved_chunks[:self.top_k])

    def get_oracle_indices(self, row) -> List[int]:
      retrieved = list(row.retrieved_chunks[:self.top_k])
      correct_idx = int(row.answer_pos)

      if correct_idx in retrieved:
          retrieved.remove(correct_idx)
      else:
          retrieved = retrieved[:-1]

      return [correct_idx] + retrieved

"""## PromptBuilder"""

class PromptBuilder:
  def __init__(self, intro="Given the following information:", command="Reply to this question:"):
    self.intro = intro
    self.command = command

  def build_baseline_prompt(self, query: str) -> str:
    return f"""
      Reply to this question in the shortest possible valid way, do not explain the answer:
      {query}
      """

  def build_zero_shot_prompt(self, query: str, chunks: List[str]) -> str:
    numbered_chunks = "\n".join(
          f"{i + 1}. {chunk}" for i, chunk in enumerate(chunks)
      )

    return self.intro + "\n" + numbered_chunks + "\n" + self.command + "\n" + query

evaluation_prompt = """
You are evaluating a generated answer for a question-answering task.

Question:
{query}

Gold short answer:
{short_answer}

Generated answer:
{generated_answer}

Evaluation criterion:
Return 1 if the generated answer contains the gold short answer in any valid form.
Valid forms include paraphrases, spelling variants, number-word equivalence, or a longer sentence that clearly contains the correct answer.
Return 0 if the generated answer does not contain the correct answer, contradicts it, gives a different entity/date/place/person, or is too vague.

Important:
- Do not reward an answer only because it is fluent.
- Do not use external knowledge.
- Judge only whether the generated answer expresses the gold short answer.
- Output only valid JSON.

Output format:
{"score": 0}
or
{"score": 1}
"""

"""## AnswerEvaluator"""

class AnswerEvaluator:

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _flatten_short_answer(short_answer):
        """
        Handles cases like:
        - ["answer"]
        - "answer"
        - ["a", "b"] → "a b"
        """
        if isinstance(short_answer, list):
            return " ".join(str(x) for x in short_answer)
        return str(short_answer)

    @staticmethod
    def get_EM(generated_answer: str, short_answer) -> int:
        generated_answer = AnswerEvaluator.normalize_text(generated_answer)
        short_answer = AnswerEvaluator.normalize_text(
            AnswerEvaluator._flatten_short_answer(short_answer)
        )
        return int(generated_answer == short_answer)

    @staticmethod
    def get_subEM(generated_answer: str, short_answer) -> int:
        generated_answer = AnswerEvaluator.normalize_text(generated_answer)
        short_answer = AnswerEvaluator.normalize_text(
            AnswerEvaluator._flatten_short_answer(short_answer)
        )
        return int(
            generated_answer in short_answer
            or short_answer in generated_answer
        )

    @staticmethod
    def get_METEOR(generated_answer: str, short_answer) -> float:
        short_answer = AnswerEvaluator._flatten_short_answer(short_answer)

        generated_tokens = word_tokenize(generated_answer)
        short_tokens = word_tokenize(short_answer)

        return meteor_score([short_tokens], generated_tokens)

    @staticmethod
    def evaluate_example(example: Dict) -> Dict:
        """
        Expects:
        {
            "generated_answer": str,
            "short_answer": str | list[str]
        }
        """

        generated_answer = example["generated_answer"]
        short_answer = example["short_answer"]

        return {
            "EM": AnswerEvaluator.get_EM(generated_answer, short_answer),
            "subEM": AnswerEvaluator.get_subEM(generated_answer, short_answer),
            "METEOR": AnswerEvaluator.get_METEOR(generated_answer, short_answer),
        }

    @staticmethod
    def evaluate_dataset(dataset: List[Dict]) -> Dict:

        em_scores = []
        subem_scores = []
        meteor_scores = []

        for example in dataset:

            result = AnswerEvaluator.evaluate_example(example)

            em_scores.append(result["EM"])
            subem_scores.append(result["subEM"])
            meteor_scores.append(result["METEOR"])

        n = len(dataset)

        return {
            "EM": sum(em_scores) / n if n else 0,
            "subEM": sum(subem_scores) / n if n else 0,
            "METEOR": sum(meteor_scores) / n if n else 0,
        }

"""## LLMJudge

## AgreementAnalyzer

## ExperimentRunner
"""

class ExperimentRunner:
  def __init__(
        self,
        prompt_builder,
        retrieval_store,
        top_k: int = 3,
        seed: int = 42,
    ):
    self.prompt_builder = prompt_builder
    self.retrieval_store = retrieval_store
    self.top_k = top_k

  def build_prompt_from_row(self, row, setting: str) -> Tuple[str, List[int]]:
    if setting == "baseline":
        prompt = self.prompt_builder.build_baseline_prompt(row.query)
        return prompt, []

    if setting == "rag":
        chunk_indices = self.retrieval_store.get_rag_indices(row)
        chunks = [row.candidate_chunks[i] for i in chunk_indices]
        prompt = self.prompt_builder.build_zero_shot_prompt(row.query, chunks)
        return prompt, chunk_indices

    if setting == "oracle":
        chunk_indices = self.retrieval_store.get_oracle_indices(row)
        chunks = [row.candidate_chunks[i] for i in chunk_indices]
        prompt = self.prompt_builder.build_zero_shot_prompt(row.query, chunks)
        return prompt, chunk_indices

  def retrieve_answers(
      self,
      df: pd.DataFrame,
      model,
      setting: str,
      thinking: bool = False,
      batch_size: int = 8,
    ):

    df = df.to_pandas() if not isinstance(df, pd.DataFrame) else df

    prompts = []
    metadata = []

    # Build prompts and metadata
    for row in df.itertuples():
        prompt, chunk_indices = self.build_prompt_from_row(row, setting)

        prompts.append(prompt)

        metadata.append({
            "query_id": row.query_id,
            "retrieved_chunks": chunk_indices,
            "augmented_prompt": prompt,
            "short_answer": getattr(row, "short_answer", None),
            "answer_pos": getattr(row, "answer_pos", None),
            "query": row.query,
        })

    results = []

    # Generate in batches
    for start in tqdm(
        range(0, len(prompts), batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc=f"{setting} generation",
    ):
        end = start + batch_size

        batch_prompts = prompts[start:end]
        batch_metadata = metadata[start:end]

        batch_answers = model.generate_answers_batch(
            batch_prompts,
            thinking=thinking,
        )

        for meta, (answer, thinking_content) in zip(batch_metadata, batch_answers):
            row_result = {
                "query_id": meta["query_id"],
                "retrieved_chunks": meta["retrieved_chunks"],
                "augmented_prompt": meta["augmented_prompt"],
                "generated_answer": answer,
            }

            # Fields for evaluation
            row_result["short_answer"] = meta["short_answer"]
            row_result["answer_pos"] = meta["answer_pos"]
            row_result["query"] = meta["query"]

            if thinking_content:
                row_result["thinking_content"] = thinking_content

            results.append(row_result)

    return results

"""## Utils"""

def export_jsonl(results: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    allowed_fields = [
        "query_id",
        "retrieved_chunks",
        "augmented_prompt",
        "generated_answer",
    ]

    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            item = {
                field: result[field]
                for field in allowed_fields
                if field in result
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def make_submission_path(
    output_dir: str,
    split: str,
    variant: str,
) -> str:
    filename = f"doubleN-{split}-{variant}.jsonl"
    return os.path.join(output_dir, filename)

def display_results_table(results: List[Dict], n: int = 5):
    preview_df = pd.DataFrame(results[:n])

    cols = [
        "query_id",
        "query",
        "retrieved_chunks",
        "generated_answer",
        "short_answer",
    ]

    cols = [c for c in cols if c in preview_df.columns]

    display(
        preview_df[cols].style.set_properties(
            **{
                "white-space": "pre-wrap",
                "text-align": "left",
                "vertical-align": "top",
            }
        )
    )

