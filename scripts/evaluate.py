import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_module import DataModule
from src.evaluation import AnswerEvaluator


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return pd.DataFrame(rows)


def get_reference_df(data_module: DataModule, split: str) -> pd.DataFrame:
    if split == "full_train":
        return data_module.to_dataframe("train").reset_index(drop=True)

    if split in {"train", "val"}:
        train_df, val_df = data_module.build_train_val_split()
        return (train_df if split == "train" else val_df).reset_index(drop=True)

    if split == "test":
        return data_module.to_dataframe("test").reset_index(drop=True)

    raise ValueError(f"Unsupported split for evaluation: {split}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_qwen3_06b_lora.yaml")
    parser.add_argument("--generated", required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val", "full_train", "test"],
        required=True,
    )
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_module = DataModule(
        dev_size=cfg["dataset"].get("dev_size", 0.2),
        seed=cfg.get("seed", 42),
    )

    data_module.load(
        url=cfg["dataset"]["url"],
        local_path=cfg["dataset"].get("local_path"),
    )

    generated_df = load_jsonl(args.generated)
    reference_df = get_reference_df(data_module, args.split)

    generated_df["query_id"] = generated_df["query_id"].astype(str)
    reference_df["query_id"] = reference_df["query_id"].astype(str)

    reference_cols = [
        "query_id",
        "query",
        "short_answer",
        "answer_pos",
    ]

    reference_cols = [
        col for col in reference_cols
        if col in reference_df.columns
    ]

    eval_df = generated_df.merge(
        reference_df[reference_cols],
        on="query_id",
        how="left",
        validate="one_to_one",
    )

    missing = eval_df["short_answer"].isna().sum()

    if missing:
        raise ValueError(
            f"Missing short_answer for {missing} generated rows. "
            f"Check that --split matches the generated file."
        )

    records = eval_df.to_dict(orient="records")

    metrics = AnswerEvaluator.evaluate_dataset(records)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()