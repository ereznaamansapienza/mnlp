

import torch
import gc
import os
import json
import numpy as np
import pandas as pd
import random
import requests

from datasets import load_dataset, Dataset, load_from_disk

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
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
generated_path = "https://raw.githubusercontent.com/ereznaamansapienza/mnlp/master/doubleN-all-test-SmolLM2-1.7B-b1-rag.jsonl"
llm_judge_output_path = "https://raw.githubusercontent.com/ereznaamansapienza/mnlp/master/doubleN-judge-subset-Qwen3-4B-v1.jsonl"

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

    def load_generated_answers_jsonl(self, path: str) -> pd.DataFrame:
        records = []

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
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_no}: {e}")
                continue

            required_fields = [
                "query_id",
                "retrieved_chunks",
                "augmented_prompt",
                "generated_answer",
            ]

            missing_fields = [
                field for field in required_fields
                if field not in obj
            ]

            if missing_fields:
                print(
                    f"Skipping line {line_no}: missing fields {missing_fields}"
                )
                continue

            records.append({
                "query_id": str(obj["query_id"]),
                "retrieved_chunks": obj["retrieved_chunks"],
                "augmented_prompt": obj["augmented_prompt"],
                "generated_answer": obj["generated_answer"],
            })

        return pd.DataFrame(records)

    def enrich_generated_answers(
        self,
        generated_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        keep_reference_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if keep_reference_columns is None:
            keep_reference_columns = [
                "query_id",
                "query",
                "short_answer",
                "answer_pos",
                "answer",
                "candidate_chunks",
            ]

        generated_df = generated_df.copy()
        reference_df = reference_df.copy()

        generated_df["query_id"] = generated_df["query_id"].astype(str)
        reference_df["query_id"] = reference_df["query_id"].astype(str)

        available_reference_columns = [
            col for col in keep_reference_columns
            if col in reference_df.columns
        ]

        reference_small_df = reference_df[available_reference_columns].copy()

        enriched_df = generated_df.merge(
            reference_small_df,
            on="query_id",
            how="left",
            validate="one_to_one",
        )

        missing_queries = enriched_df["query"].isna().sum()

        if missing_queries > 0:
            print(f"Warning: {missing_queries} generated rows could not be matched to reference_df.")

        return enriched_df

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

evaluation_prompt = """###Task Description:
You are evaluating a generated answer for a question-answering task.

Return only one character:
0 or 1

###The instruction to evaluate:
{query}

###Response to evaluate:
{generated_answer}

###Reference Answer (Score 1):
{short_answer}

###Score Rubrics:
Score 0: The generated answer does not contain the correct answer, contradicts it, gives a different entity/date/place/person, or is too vague.
Score 1: The generated answer contains the gold short answer in any valid form. alid forms include paraphrases, spelling variants, number-word equivalence, or a longer sentence that clearly contains the correct answer.
"""

training_prompt = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

inference_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>

"""

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

  def build_evaluation_prompt(self, query: str, short_answer: str, generated_answer: str) -> str:
    return evaluation_prompt.format(
        query=query,
        generated_answer=generated_answer,
        short_answer=short_answer,
    )

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

"""## LLMJudge"""

class LLMJudge(LMGenerator):
    def __init__(
        self,
        model_name: str = "prometheus-eval/prometheus-7b-v2.0",
        max_new_tokens: int = 512,
        supports_thinking: bool = False
    ):
        super().__init__(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            supports_thinking=supports_thinking,
        )

    def format_prompt(self, prompt: str, thinking: bool = False) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def parse_score(output: str):
        output = str(output).strip()

        if not output:
            return None

        # Normalize common full-width digits just in case.
        output = output.replace("０", "0").replace("１", "1")

        # Best case: exactly one character.
        if output in {"0", "1"}:
            return int(output)

        # Common structured outputs:
        # [RESULT] 1
        # RESULT: 1
        # Score: 1
        # score = 0
        patterns = [
            r"\[?\s*RESULT\s*\]?\s*[:=\-]?\s*([01])\b",
            r"\bscore\s*[:=\-]?\s*([01])\b",
            r"\bfinal\s*score\s*[:=\-]?\s*([01])\b",
            r"\banswer\s*[:=\-]?\s*([01])\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, output, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))

        # If the model outputs something like:
        # "1."
        # "(1)"
        # "`1`"
        cleaned = output.strip().strip("`'\".()[]{} \n\t")
        if cleaned in {"0", "1"}:
            return int(cleaned)

        # Last fallback: only accept if there is exactly one standalone 0/1.
        matches = re.findall(r"(?<!\d)[01](?!\d)", output)
        if len(matches) == 1:
            return int(matches[0])

        return None

    def judge_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 4,
        keep_output: bool = False,
        thinking_mode: bool = False,
    ) -> pd.DataFrame:
        df = df.copy()

        prompts = [
            evaluation_prompt.format(
                query=row.query,
                short_answer=row.short_answer,
                generated_answer=row.generated_answer,
            )
            for row in df.itertuples()
        ]

        scores = []
        raw_outputs = []

        for start in tqdm(range(0, len(prompts), batch_size), desc="LLM judging"):
            batch_prompts = prompts[start:start + batch_size]
            batch_outputs = self.generate_answers_batch(batch_prompts, thinking=thinking_mode)

            for output, _ in batch_outputs:
                scores.append(self.parse_score(output))

                if keep_output:
                    raw_outputs.append(output)

        df["llm_judge_a"] = scores

        if keep_output:
            df["llm_judge_output"] = raw_outputs

        return df


"""## SFT Dataset Builder"""

class SFTDatasetBuilder:
    def __init__(
        self,
        prompt_builder: PromptBuilder,
        top_k: int = 3,
        seed: int = 42,
    ):
        self.prompt_builder = prompt_builder
        self.top_k = top_k
        self.seed = seed
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    #  Chunk selectors                                                     #
    # ------------------------------------------------------------------ #

    def _get_distractor_indices(self, row, exclude_idx: int) -> List[int]:
        """All candidate indices except the correct one, in original order."""
        return [
            i for i in range(len(row.candidate_chunks))
            if i != exclude_idx
        ]

    def _get_oracle_first_chunks(self, row) -> List[str]:
        """Correct chunk first, then top_k-1 distractors."""
        correct_idx = int(row.answer_pos)
        distractors = self._get_distractor_indices(row, correct_idx)
        selected = [correct_idx] + distractors[:self.top_k - 1]
        return [row.candidate_chunks[i] for i in selected]

    def _get_oracle_random_chunks(self, row) -> List[str]:
        """Correct chunk at a random position among top_k chunks."""
        chunks = self._get_oracle_first_chunks(row)
        self.rng.shuffle(chunks)
        return chunks

    def _get_no_answer_chunks(self, row) -> List[str]:
        """Only distractors — correct chunk deliberately excluded."""
        correct_idx = int(row.answer_pos)
        distractors = self._get_distractor_indices(row, correct_idx)
        # sample randomly so no_answer examples are diverse
        sampled = self.rng.sample(
            distractors,
            k=min(self.top_k, len(distractors)),
        )
        return [row.candidate_chunks[i] for i in sampled]

    # ------------------------------------------------------------------ #
    #  Response builders                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_answer(row) -> str:
        answer = row.short_answer
        if isinstance(answer, list):
            answer = " ".join(str(x) for x in answer)
        return str(answer).strip()

    @staticmethod
    def _get_no_answer_response() -> str:
        return "I don't have that information."

    # ------------------------------------------------------------------ #
    #  Single-example builder                                              #
    # ------------------------------------------------------------------ #

    def _build_example(self, row, chunks: List[str], response: str, variant: str) -> Dict:
        prompt = self.prompt_builder.build_zero_shot_prompt(row.query, chunks)
        return {
            "query_id": str(row.query_id),
            "prompt":   prompt,
            "response": response,
            "variant":  variant,
        }

    # ------------------------------------------------------------------ #
    #  DataFrame → SFT examples                                           #
    # ------------------------------------------------------------------ #

    def build_sft_examples(
        self,
        df: pd.DataFrame,
        variants: List[str] = ("oracle_first", "oracle_random", "no_answer"),
        no_answer_ratio: float = 0.3,
    ) -> List[Dict]:
        """
        Builds training examples directly from candidate_chunks and answer_pos.

        Variants:
          oracle_first  — correct chunk placed first  → gold answer
          oracle_random — correct chunk shuffled in   → gold answer
          no_answer     — only distractor chunks      → abstention response
        """
        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()

        # sample rows for no_answer variant
        no_answer_ids = set(
            df.sample(frac=no_answer_ratio, random_state=self.seed)["query_id"]
            .astype(str)
            .tolist()
        )

        examples = []

        for row in df.itertuples():
            q_id = str(row.query_id)

            if "oracle_first" in variants:
                examples.append(self._build_example(
                    row,
                    self._get_oracle_first_chunks(row),
                    self._get_answer(row),
                    "oracle_first",
                ))

            if "oracle_random" in variants:
                examples.append(self._build_example(
                    row,
                    self._get_oracle_random_chunks(row),
                    self._get_answer(row),
                    "oracle_random",
                ))

            if "no_answer" in variants and q_id in no_answer_ids:
                examples.append(self._build_example(
                    row,
                    self._get_no_answer_chunks(row),
                    self._get_no_answer_response(),
                    "no_answer",
                ))

        self.rng.shuffle(examples)
        return examples

    # ------------------------------------------------------------------ #
    #  HuggingFace Dataset                                                 #
    # ------------------------------------------------------------------ #

    def build_hf_dataset(
        self,
        df: pd.DataFrame,
        tokenizer,
        variants: List[str] = ("oracle_first", "oracle_random", "no_answer"),
        no_answer_ratio: float = 0.3,
    ) -> Dataset:
        examples = self.build_sft_examples(df, variants, no_answer_ratio)

        texts = []
        for ex in examples:
            messages = [
                {"role": "user",      "content": ex["prompt"]},
                {"role": "assistant", "content": ex["response"]},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        return Dataset.from_dict({"text": texts})


"""## LLMFineTuner"""

class LLMFineTuner:
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        # LoRA
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: List[str] = None,
        # Training
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.03,
        max_seq_length: int = 512,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.seed = seed

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules or ["q_proj", "v_proj"],
        )

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            seed=seed,
        )

    def load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"   # right for SFT

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()

        return model, tokenizer

    def finetune(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
    ):
        model, tokenizer = self.load_model_and_tokenizer()

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=self.training_args,
            max_seq_length=self.max_seq_length,
            dataset_text_field="text",
        )

        trainer.train()

        best_model_dir = os.path.join(self.output_dir, "best_model")
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        print(f"Model saved to {best_model_dir}")

        return model, tokenizer, trainer


"""## AgreementAnalyzer

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

def export_jsonl(
    results,
    path: str,
    include_eval: bool = False,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    if isinstance(results, pd.DataFrame):
        results = results.to_dict(orient="records")

    allowed_fields = [
        "query_id",
        "retrieved_chunks",
        "augmented_prompt",
        "generated_answer",
    ]

    if include_eval:
        allowed_fields += [
            "llm_judge_a",
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

import pandas as pd
import os

def make_manual_annotation_csv(
    judge_jsonl_path: str,
    reference_df: pd.DataFrame,
    output_csv_path: str,
):
    # Load your JSONL with generated answers + llm_judge_a
    judge_df = pd.read_json(judge_jsonl_path, lines=True)

    # Attach query / short_answer if missing
    reference_small = reference_df[[
        "query_id",
        "query",
        "short_answer",
    ]].copy()

    judge_df["query_id"] = judge_df["query_id"].astype(str)
    reference_small["query_id"] = reference_small["query_id"].astype(str)

    judge_df = judge_df.merge(
        reference_small,
        on="query_id",
        how="left",
        validate="one_to_one",
    )

    # Keep only annotation-relevant fields
    manual_df = judge_df[[
        "query_id",
        "query",
        "short_answer",
        "generated_answer",
        "llm_judge_a",
    ]].copy()

    manual_df["annotator_1"] = ""
    manual_df["annotator_2"] = ""

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    manual_df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    return manual_df

