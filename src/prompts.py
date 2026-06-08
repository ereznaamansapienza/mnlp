from dataclasses import dataclass
from typing import Iterable, List


EVALUATION_PROMPT = """###Task Description:
You are evaluating a generated answer for a question-answering task.

Return only one character:
0 or 1

###Question:
{query}

###Generated answer:
{generated_answer}

###Gold short answer:
{short_answer}

###Score Rubric:
Score 0: The generated answer does not contain the correct answer, contradicts it, gives a different entity/date/place/person, or is too vague.
Score 1: The generated answer contains the gold short answer in any valid form. Valid forms include paraphrases, spelling variants, number-word equivalence, or a longer sentence that clearly contains the correct answer.

Output only 0 or 1.
"""


@dataclass
class PromptBuilder:
    intro: str = "Given the following information:"
    command: str = "Reply to this question:"
    short_answer_instruction: str = "Answer with the shortest valid answer only. Do not explain."

    def format_chunks(self, chunks: Iterable[str]) -> str:
        return "\n".join(f"{i + 1}. {chunk}" for i, chunk in enumerate(chunks))

    def build_baseline_prompt(self, query: str) -> str:
        return f"""Reply to this question in the shortest possible valid way.
Do not explain.

Question:
{query}

Answer:"""

    def build_rag_prompt(self, query: str, chunks: List[str]) -> str:
        numbered_chunks = self.format_chunks(chunks)
        return f"""{self.intro}
{numbered_chunks}

{self.command}
{query}

{self.short_answer_instruction}

Answer:"""

    # Backwards-compatible alias used in the previous notebook.
    def build_zero_shot_prompt(self, query: str, chunks: List[str]) -> str:
        return self.build_rag_prompt(query, chunks)

    def build_evaluation_prompt(self, query: str, short_answer: str, generated_answer: str) -> str:
        return EVALUATION_PROMPT.format(
            query=query,
            generated_answer=generated_answer,
            short_answer=short_answer,
        )
