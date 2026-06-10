import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import RetrievalStore
from src.data_module import DataModule, SFTDatasetBuilder
from src.prompts import PromptBuilder
from src.training import train_lora
from src.utils import set_seed
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_qwen3_06b_lora.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    data_module = DataModule(
        dev_size=cfg["dataset"].get("dev_size", 0.2),
        seed=cfg.get("seed", 42),
    )
    data_module.load(
        url=cfg["dataset"]["url"],
        local_path=cfg["dataset"].get("local_path"),
    )
    train_df, val_df = data_module.build_train_val_split()

    retrieval_cfg = cfg["retrieval"]

    retrieval_store = RetrievalStore(top_k=retrieval_cfg["top_k"])

    train_ranking_path = retrieval_cfg.get("train_ranking_path")

    if train_ranking_path:
        train_df = retrieval_store.attach_rankings_from_jsonl(
            train_df,
            train_ranking_path,
        )

        val_df = retrieval_store.attach_rankings_from_jsonl(
            val_df,
            train_ranking_path,
    )

    prompt_builder = PromptBuilder()
    builder = SFTDatasetBuilder(
        prompt_builder=prompt_builder,
        top_k=cfg["sft_data"].get("top_k", 3),
        seed=cfg.get("seed", 42),
    )

    variants = cfg["sft_data"].get("variants", ["oracle_first", "oracle_random", "no_answer"])
    train_dataset = builder.build_hf_dataset(
        train_df,
        variants=variants,
        no_answer_ratio=cfg["sft_data"].get("no_answer_ratio", 0.3),
    )
    val_dataset = builder.build_hf_dataset(
        val_df,
        variants=variants,
        no_answer_ratio=cfg["sft_data"].get("no_answer_ratio", 0.3),
    )

    print("Train dataset size:", len(train_dataset))
    print("Train variants:", Counter(train_dataset["variant"]))

    print("Val dataset size:", len(val_dataset))
    print("Val variants:", Counter(val_dataset["variant"]))

    final_dir = train_lora(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=cfg["model"]["name"],
        output_dir=cfg["training"]["output_dir"],
        max_seq_length=cfg["model"].get("max_seq_length", 1024),
        load_in_4bit=cfg["model"].get("load_in_4bit", True),
        lora_r=cfg["training"].get("lora_r", 32),
        lora_alpha=cfg["training"].get("lora_alpha", 64),
        lora_dropout=cfg["training"].get("lora_dropout", 0.05),
        num_train_epochs=cfg["training"].get("num_train_epochs", 1),
        per_device_train_batch_size=cfg["training"].get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg["training"].get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 4),
        learning_rate=cfg["training"].get("learning_rate", 2e-4),
        warmup_steps=cfg["training"].get("warmup_steps", 10),
        logging_steps=cfg["training"].get("logging_steps", 10),
        save_steps=cfg["training"].get("save_steps", 200),
        seed=cfg.get("seed", 42),
    )
    print(f"Saved final LoRA adapter to: {final_dir}")


if __name__ == "__main__":
    main()
