import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}, line {line_no}: {exc}") from exc
    return rows


def write_jsonl(rows: Iterable[Mapping], path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def export_generation_jsonl(results, path: str, include_eval: bool = False) -> None:
    if isinstance(results, pd.DataFrame):
        records = results.to_dict(orient="records")
    else:
        records = list(results)

    allowed_fields = [
        "query_id",
        "retrieved_chunks",
        "augmented_prompt",
        "generated_answer",
    ]
    if include_eval:
        allowed_fields.append("llm_judge_a")

    output = []
    for record in records:
        output.append({field: record[field] for field in allowed_fields if field in record})
    write_jsonl(output, path)


def make_submission_path(output_dir: str, split: str, variant: str) -> str:
    return os.path.join(output_dir, "submissions", f"doubleN-{split}-{variant}.jsonl")
