import re
import string
from typing import Dict, List

import pandas as pd
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize


class AnswerEvaluator:
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
        for row in df.itertuples():
            scores.append({
                "EM": cls.em(row.generated_answer, row.short_answer),
                "subEM": cls.subem(row.generated_answer, row.short_answer),
                "METEOR": cls.meteor(row.generated_answer, row.short_answer),
            })
        if not scores:
            return {"EM": 0.0, "subEM": 0.0, "METEOR": 0.0}
        return pd.DataFrame(scores).mean().to_dict()


def add_automatic_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EM"] = [AnswerEvaluator.em(r.generated_answer, r.short_answer) for r in df.itertuples()]
    df["subEM"] = [AnswerEvaluator.subem(r.generated_answer, r.short_answer) for r in df.itertuples()]
    df["METEOR"] = [AnswerEvaluator.meteor(r.generated_answer, r.short_answer) for r in df.itertuples()]
    return df
