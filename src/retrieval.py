import json
import os
from typing import Dict, List, Optional

import pandas as pd
import requests


def github_blob_to_raw_url(url: str) -> str:
    if "github.com" in url and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        url = url.replace("/blob/", "/")
    return url


class RetrievalStore:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def _read_lines(self, path: str) -> List[str]:
        path = github_blob_to_raw_url(path)
        if path.startswith("http://") or path.startswith("https://"):
            response = requests.get(path, timeout=60)
            response.raise_for_status()
            text = response.text
            if text.lstrip().startswith("<"):
                raise ValueError("Downloaded HTML, not JSONL. Use a raw GitHub URL.")
            return text.splitlines()
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()

    def load_rankings_jsonl(self, path: str) -> Dict[str, List[int]]:
        rankings: Dict[str, List[int]] = {}
        for line_no, line in enumerate(self._read_lines(path), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid ranking JSONL at line {line_no}: {exc}") from exc
            query_id, ranking = next(iter(obj.items()))
            rankings[str(query_id)] = [int(i) for i in ranking]
        return rankings

    def attach_rankings_from_jsonl(self, df: pd.DataFrame, path: str) -> pd.DataFrame:
        df = df.copy()
        rankings = self.load_rankings_jsonl(path)
        df["ranking"] = df["query_id"].astype(str).map(rankings)

        missing = df[df["ranking"].isna()]["query_id"].astype(str).tolist()
        if missing:
            raise ValueError(
                f"Missing rankings for {len(missing)} query_ids. First examples: {missing[:5]}. "
                "Do not use test rankings for train rows; query IDs will not match."
            )

        df["retrieved_chunks"] = df["ranking"].apply(lambda ranking: list(ranking[: self.top_k]))
        self.validate_indices(df, column="retrieved_chunks")
        return df

    def validate_indices(self, df: pd.DataFrame, column: str = "retrieved_chunks") -> None:
        for row in df.itertuples():
            n_candidates = len(row.candidate_chunks)
            for idx in getattr(row, column):
                if int(idx) < 0 or int(idx) >= n_candidates:
                    raise ValueError(
                        f"Invalid chunk index {idx} for query_id={row.query_id}. n_candidates={n_candidates}"
                    )

    def load_generated_answers_jsonl(self, path: str) -> pd.DataFrame:
        rows = []
        for line_no, line in enumerate(self._read_lines(path), start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "query_id": str(obj["query_id"]),
                "retrieved_chunks": obj.get("retrieved_chunks", []),
                "augmented_prompt": obj.get("augmented_prompt", ""),
                "generated_answer": obj.get("generated_answer", ""),
                **({"llm_judge_a": obj["llm_judge_a"]} if "llm_judge_a" in obj else {}),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No valid generated answers were loaded.")
        return df

    def enrich_generated_answers(
        self,
        generated_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        keep_reference_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if keep_reference_columns is None:
            keep_reference_columns = ["query_id", "query", "short_answer", "answer_pos", "answer", "candidate_chunks"]
        generated_df = generated_df.copy()
        reference_df = reference_df.copy()
        generated_df["query_id"] = generated_df["query_id"].astype(str)
        reference_df["query_id"] = reference_df["query_id"].astype(str)
        cols = [c for c in keep_reference_columns if c in reference_df.columns]
        merged = generated_df.merge(reference_df[cols], on="query_id", how="left", validate="one_to_one")
        if "query" in merged.columns and merged["query"].isna().any():
            missing = int(merged["query"].isna().sum())
            raise ValueError(f"Could not match {missing} generated rows to reference_df.")
        return merged

    def get_rag_indices(self, row) -> List[int]:
        return list(row.retrieved_chunks[: self.top_k])

    def get_oracle_indices(self, row) -> List[int]:
        retrieved = list(row.retrieved_chunks[: self.top_k]) if hasattr(row, "retrieved_chunks") else []
        correct_idx = int(row.answer_pos)
        if correct_idx in retrieved:
            retrieved.remove(correct_idx)
        else:
            retrieved = retrieved[: max(0, self.top_k - 1)]
        return [correct_idx] + retrieved[: self.top_k - 1]
