import argparse
import json
import sys
from pathlib import Path

import nltk
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_module import DataModule
from src.evaluation import AnswerEvaluator, add_automatic_scores
from src.retrieval import RetrievalStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_qwen3_06b_lora.yaml")
    parser.add_argument("--generated", required=True, help="Generated test JSONL path")
    parser.add_argument("--out", default="outputs/eval_metrics.json")
    args = parser.parse_args()

    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_module = DataModule(seed=cfg.get("seed", 42))
    data_module.load(url=cfg["dataset"]["url"])
    test_df = data_module.to_dataframe("test")

    store = RetrievalStore(top_k=cfg["sft_data"].get("top_k", 3))
    generated_df = store.load_generated_answers_jsonl(args.generated)
    eval_df = store.enrich_generated_answers(generated_df, test_df)
    eval_df = add_automatic_scores(eval_df)
    metrics = AnswerEvaluator.evaluate_dataframe(eval_df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
