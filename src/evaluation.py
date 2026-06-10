import re
import string
from typing import Dict, List

import pandas as pd
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

import os
import nltk
NLTK_DATA_DIR = os.environ.get(
    "NLTK_DATA",
    "/leonardo_work/IscrC_MNLP26/apromots_doubleN/nltk_data",
)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)


class AnswerEvaluator:
    NO_ANSWER_PHRASES = [
    "i dont have that information",
    "i do not have that information",
    "i dont know",
    "i do not know",
    "not enough information",
    "insufficient information",
    "cannot answer",
    "cant answer",
    "unknown",
    ]

    @classmethod
    def is_no_answer(cls, generated_answer: str) -> int:
        generated = cls.normalize_text(generated_answer)
        return int(any(phrase in generated for phrase in cls.NO_ANSWER_PHRASES))

    @classmethod
    def should_no_answer(cls, row) -> int:
        if not hasattr(row, "retrieved_chunks") or not hasattr(row, "answer_pos"):
            return 0

        retrieved = row.retrieved_chunks

        if retrieved is None:
            return 0

        if isinstance(retrieved, str):
            import ast
            retrieved = ast.literal_eval(retrieved)

        retrieved = [int(x) for x in retrieved]
        correct_idx = int(row.answer_pos)

        # If the official gold chunk is retrieved, answer is supported.
        if correct_idx in retrieved:
            return 0

        # If another retrieved chunk literally contains the gold short answer,
        # treat it as answer-supported too.
        if hasattr(row, "candidate_chunks") and hasattr(row, "short_answer"):
            answer = cls.normalize_text(cls.flatten_answer(row.short_answer))

            chunks = [
                row.candidate_chunks[i]
                for i in retrieved
                if 0 <= i < len(row.candidate_chunks)
            ]

            joined = cls.normalize_text(" ".join(str(chunk) for chunk in chunks))

            if answer and answer in joined:
                return 0

        return 1

    @staticmethod
    def binary_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return {
            "NO_ANSWER_PRECISION": precision,
            "NO_ANSWER_RECALL": recall,
            "NO_ANSWER_F1": f1,
        }

    @staticmethod
    def normalize_text(text: str) -> str:
        text = str(text).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def flatten_answer(short_answer) -> str:
        if isinstance(short_answer, list):
            return " ".join(str(x) for x in short_answer)
        return str(short_answer)

    @classmethod
    def em(cls, generated_answer: str, short_answer) -> int:
        return int(cls.normalize_text(generated_answer) == cls.normalize_text(cls.flatten_answer(short_answer)))

    @classmethod
    def subem(cls, generated_answer: str, short_answer) -> int:
        generated = cls.normalize_text(generated_answer)
        gold = cls.normalize_text(cls.flatten_answer(short_answer))
        return int(gold in generated or generated in gold)

    @classmethod
    def meteor(cls, generated_answer: str, short_answer) -> float:
        generated_tokens = word_tokenize(str(generated_answer))
        gold_tokens = word_tokenize(cls.flatten_answer(short_answer))
        return float(meteor_score([gold_tokens], generated_tokens))

    @classmethod
    def evaluate_dataframe(cls, df: pd.DataFrame) -> Dict[str, float]:
        scores = []

        no_answer_true = []
        no_answer_pred = []

        for row in df.itertuples():
            scores.append({
                "EM": cls.em(row.generated_answer, row.short_answer),
                "subEM": cls.subem(row.generated_answer, row.short_answer),
                "METEOR": cls.meteor(row.generated_answer, row.short_answer),
            })

            should_na = cls.should_no_answer(row)
            pred_na = cls.is_no_answer(row.generated_answer)

            no_answer_true.append(should_na)
            no_answer_pred.append(pred_na)

        if not scores:
            return {
                "EM": 0.0,
                "subEM": 0.0,
                "METEOR": 0.0,
                "NO_ANSWER_PRECISION": 0.0,
                "NO_ANSWER_RECALL": 0.0,
                "NO_ANSWER_F1": 0.0,
                "NO_ANSWER_RATE": 0.0,
                "SHOULD_NO_ANSWER_RATE": 0.0,
            }

        metrics = pd.DataFrame(scores).mean().to_dict()

        tp = sum(1 for y, p in zip(no_answer_true, no_answer_pred) if y == 1 and p == 1)
        fp = sum(1 for y, p in zip(no_answer_true, no_answer_pred) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(no_answer_true, no_answer_pred) if y == 1 and p == 0)

        metrics.update(cls.binary_prf(tp, fp, fn))
        metrics["NO_ANSWER_RATE"] = sum(no_answer_pred) / len(no_answer_pred)
        metrics["SHOULD_NO_ANSWER_RATE"] = sum(no_answer_true) / len(no_answer_true)

        return metrics


def add_automatic_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EM"] = [AnswerEvaluator.em(r.generated_answer, r.short_answer) for r in df.itertuples()]
    df["subEM"] = [AnswerEvaluator.subem(r.generated_answer, r.short_answer) for r in df.itertuples()]
    df["METEOR"] = [AnswerEvaluator.meteor(r.generated_answer, r.short_answer) for r in df.itertuples()]
    df["pred_no_answer"] = [AnswerEvaluator.is_no_answer(r.generated_answer) for r in df.itertuples()]
    df["should_no_answer"] = [AnswerEvaluator.should_no_answer(r) for r in df.itertuples()]
    return df
