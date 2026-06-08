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


def generate_split(df, split_name, model, runner, cfg):
    results = runner.retrieve_answers(
        df=df,
        model=model,
        setting="rag",
        batch_size=cfg["generation"].get("batch_size", 8),
        thinking=False,
    )
    out = Path(cfg["generation"]["output_dir"]) / f"doubleN-{split_name}-{cfg['generation']['variant_name']}.jsonl"
    export_generation_jsonl(results, str(out))
    print(f"Saved {split_name}: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_qwen3_06b_lora.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    data_module = DataModule(seed=cfg.get("seed", 42))
    data_module.load(url=cfg["dataset"]["url"])
    test_df = data_module.to_dataframe("test")
    blind_df = data_module.to_dataframe("blind")

    store = RetrievalStore(top_k=cfg["sft_data"].get("top_k", 3))
    test_df = store.attach_rankings_from_jsonl(test_df, cfg["generation"]["test_ranking_path"])
    blind_df = store.attach_rankings_from_jsonl(blind_df, cfg["generation"]["blind_ranking_path"])

    model = LMGenerator(
        model_name=cfg["model"]["name"],
        adapter_path=cfg["generation"].get("adapter_path"),
        max_new_tokens=cfg["generation"].get("max_new_tokens", 32),
    )
    runner = ExperimentRunner(PromptBuilder(), store, top_k=cfg["sft_data"].get("top_k", 3))

    generate_split(test_df, "test", model, runner, cfg)
    generate_split(blind_df, "blind", model, runner, cfg)


if __name__ == "__main__":
    main()
