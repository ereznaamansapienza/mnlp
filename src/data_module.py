import random
from typing import Dict, List, Tuple

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from sklearn.model_selection import train_test_split

from .prompts import PromptBuilder


class DataModule:
    def __init__(self, dev_size: float = 0.2, seed: int = 42):
        self.ds = None
        self.dev_size = dev_size
        self.seed = seed

    def load(self, url: str | None = None, local_path: str | None = None):
        if local_path:
            self.ds = load_from_disk(local_path)
        elif url:
            self.ds = load_dataset(url)
        else:
            raise ValueError("Either url or local_path must be provided.")
        return self.ds

    def to_dataframe(self, split_name: str) -> pd.DataFrame:
        if self.ds is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return pd.DataFrame(self.ds[split_name]).reset_index(drop=True)

    def build_train_val_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df_full = self.to_dataframe("train")
        train_df, val_df = train_test_split(
            train_df_full,
            test_size=self.dev_size,
            random_state=self.seed,
            shuffle=True,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class SFTDatasetBuilder:
    def __init__(self, prompt_builder: PromptBuilder, top_k: int = 3, seed: int = 42):
        self.prompt_builder = prompt_builder
        self.top_k = top_k
        self.seed = seed
        self.rng = random.Random(seed)

    @staticmethod
    def _answer_text(row) -> str:
        answer = row.short_answer
        if isinstance(answer, list):
            answer = " ".join(str(x) for x in answer)
        return str(answer).strip()

    @staticmethod
    def _no_answer_text() -> str:
        return "I don't have that information."

    def _distractor_indices(self, row, exclude_idx: int) -> List[int]:
        return [i for i in range(len(row.candidate_chunks)) if i != exclude_idx]

    def _chunks_from_indices(self, row, indices: List[int]) -> List[str]:
        return [row.candidate_chunks[int(i)] for i in indices[: self.top_k]]

    def _oracle_first_indices(self, row) -> List[int]:
        correct_idx = int(row.answer_pos)
        distractors = self._distractor_indices(row, correct_idx)
        return [correct_idx] + distractors[: self.top_k - 1]

    def _oracle_random_indices(self, row) -> List[int]:
        indices = self._oracle_first_indices(row)
        self.rng.shuffle(indices)
        return indices

    def _rag_retrieved_indices(self, row) -> List[int]:
        if not hasattr(row, "retrieved_chunks"):
            raise ValueError("rag_retrieved variant requires a retrieved_chunks column.")
        return list(row.retrieved_chunks[: self.top_k])

    def _rag_retrieved_contains_answer(self, row) -> bool:
        if not hasattr(row, "retrieved_chunks"):
            return False

        correct_idx = int(row.answer_pos)
        retrieved = list(row.retrieved_chunks[:self.top_k])

        return correct_idx in retrieved

    def _build_example(self, row, indices: List[int], response: str, variant: str) -> Dict:
        chunks = self._chunks_from_indices(row, indices)
        prompt = self.prompt_builder.build_rag_prompt(row.query, chunks)
        return {
            "query_id": str(row.query_id),
            "prompt": prompt,
            "response": response,
            "retrieved_chunks": [int(i) for i in indices],
            "variant": variant,
        }

    def _chunks_contain_answer_text(self, row, indices: List[int]) -> bool:
        answer = self._answer_text(row).lower()
        chunks = self._chunks_from_indices(row, indices)
        joined = " ".join(str(chunk) for chunk in chunks).lower()
        return answer in joined


    def _no_answer_indices(self, row) -> List[int]:
        correct_idx = int(row.answer_pos)
        answer = self._answer_text(row).lower()

        distractors = [
            i for i in range(len(row.candidate_chunks))
            if i != correct_idx
            and answer not in str(row.candidate_chunks[i]).lower()
        ]

        k = min(self.top_k, len(distractors))

        if k == 0:
            return []

        return self.rng.sample(distractors, k=k)


    def build_examples(
        self,
        df: pd.DataFrame,
        variants: List[str],
        no_answer_ratio: float = 0.3,
    ) -> List[Dict]:
        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()

        df = df.reset_index(drop=True)

        no_answer_ids = set()
        if "no_answer" in variants and no_answer_ratio > 0:
            no_answer_ids = set(
                df.sample(
                    frac=no_answer_ratio,
                    random_state=self.seed,
                )["query_id"].astype(str).tolist()
            )

        examples: List[Dict] = []

        for row in df.itertuples():
            qid = str(row.query_id)

            if "rag_retrieved" in variants:
                indices = self._rag_retrieved_indices(row)

                if (
                    self._rag_retrieved_contains_answer(row)
                    or self._chunks_contain_answer_text(row, indices)
                ):
                    response = self._answer_text(row)
                else:
                    response = self._no_answer_text()

                examples.append(
                    self._build_example(
                        row,
                        indices,
                        response,
                        "rag_retrieved",
                    )
                )

            if "oracle_first" in variants:
                examples.append(
                    self._build_example(
                        row,
                        self._oracle_first_indices(row),
                        self._answer_text(row),
                        "oracle_first",
                    )
                )

            if "oracle_random" in variants:
                examples.append(
                    self._build_example(
                        row,
                        self._oracle_random_indices(row),
                        self._answer_text(row),
                        "oracle_random",
                    )
                )

            if "no_answer" in variants and qid in no_answer_ids:
                indices = self._no_answer_indices(row)

                if len(indices) == self.top_k:
                    examples.append(
                        self._build_example(
                            row,
                            indices,
                            self._no_answer_text(),
                            "no_answer",
                        )
                    )

        self.rng.shuffle(examples)
        return examples

    def build_hf_dataset(self, df: pd.DataFrame, variants: List[str], no_answer_ratio: float = 0.3) -> Dataset:
        examples = self.build_examples(df, variants=variants, no_answer_ratio=no_answer_ratio)
        return Dataset.from_list(examples)
