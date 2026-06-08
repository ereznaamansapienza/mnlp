import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_module import DataModule
from src.generation import ExperimentRunner, LMGenerator
from src.prompts import PromptBuilder
from src.retrieval import RetrievalStore
from src.utils import export_generation_jsonl, set_seed


def get_split_df(data_module: DataModule, split: str):
    """
    Returns the requested split as a DataFrame.

    train / val are recreated from the original train split using the same seed.
    full_train uses the full original train split.
    test / blind use the official dataset splits.
    """
    if split == "full_train":
        return data_module.to_dataframe("train").reset_index(drop=True)

    if split in {"train", "val"}:
        train_df, val_df = data_module.build_train_val_split()
        df = train_df if split == "train" else val_df
        return df.reset_index(drop=True)

    if split in {"test", "blind"}:
        return data_module.to_dataframe(split).reset_index(drop=True)

    raise ValueError(f"Unknown split: {split}")


def get_ranking_path(cfg, split: str):
    retrieval_cfg = cfg["retrieval"]

    if split in {"train", "val", "full_train"}:
        return retrieval_cfg["train_ranking_path"]

    if split == "test":
        return retrieval_cfg["test_ranking_path"]

    if split == "blind":
        return retrieval_cfg["blind_ranking_path"]

    raise ValueError(f"Unknown split: {split}")


def generate_split(df, split_name, model, runner, cfg, setting: str):
    results = runner.retrieve_answers(
        df=df,
        model=model,
        setting=setting,
        batch_size=cfg["generation"].get("batch_size", 8),
        thinking=False,
    )

    out_dir = Path(cfg["generation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_name = cfg["generation"]["variant_name"]
    out = out_dir / f"doubleN-{split_name}-{variant_name}-{setting}.jsonl"

    export_generation_jsonl(results, str(out))

    print(f"Saved {split_name}: {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_qwen3_06b_lora.yaml")
    parser.add_argument(
        "--split",
        choices=["train", "val", "full_train", "test", "blind"],
        required=True,
    )
    parser.add_argument(
        "--setting",
        choices=["baseline", "rag", "oracle"],
        default="rag",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    data_module = DataModule(
        dev_size=cfg["dataset"].get("dev_size", 0.2),
        seed=seed,
    )

    data_module.load(
        url=cfg["dataset"]["url"],
        local_path=cfg["dataset"].get("local_path"),
    )

    df = get_split_df(data_module, args.split)

    store = RetrievalStore(top_k=cfg["retrieval"].get("top_k", 3))

    ranking_path = get_ranking_path(cfg, args.split)
    df = store.attach_rankings_from_jsonl(df, ranking_path)

    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)

    model = LMGenerator(
        model_name=cfg["model"]["name"],
        adapter_path=cfg["generation"].get("adapter_path"),
        max_new_tokens=cfg["generation"].get("max_new_tokens", 32),
    )

    runner = ExperimentRunner(
        PromptBuilder(),
        store,
        top_k=cfg["retrieval"].get("top_k", 3),
    )

    generate_split(
        df=df,
        split_name=args.split,
        model=model,
        runner=runner,
        cfg=cfg,
        setting=args.setting,
    )


if __name__ == "__main__":
    main()
