# MNLP HW2 RAG Fine-tuning Pipeline

This package is a cleaned-up version of the notebook code split into reusable files.

## Objective

Fine-tune `Qwen/Qwen3-0.6B` with LoRA/QLoRA for the HW2 RAG answer-generation task:

```text
retrieved chunks + question -> short answer
```

The SFT data builder creates RAG-style examples from the training split using:

- `oracle_first`: correct chunk first + distractors -> gold answer
- `oracle_random`: correct chunk shuffled among distractors -> gold answer
- `no_answer`: only distractor chunks -> abstention answer

This avoids the incorrect pattern of attaching **test rankings** to the **train split**.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train_lora.py --config configs/train_qwen3_06b_lora.yaml
```

Output adapter:

```text
outputs/qwen3_06b_rag_lora/final_adapter
```

## Generate test/blind RAG answers

```bash
python scripts/generate_test.py --config configs/train_qwen3_06b_lora.yaml
```

Outputs are saved under:

```text
outputs/submissions/
```

## Evaluate generated test file

```bash
python scripts/evaluate.py \
  --config configs/train_qwen3_06b_lora.yaml \
  --generated outputs/submissions/doubleN-test-qwen3-0.6b-rag-lora.jsonl \
  --out outputs/eval_metrics.json
```

## Notes

- Qwen3 thinking mode is disabled in chat-template formatting for short-answer RAG.
- The training code masks the user prompt tokens with `-100`, so the loss is applied mainly to the assistant answer.
- Generation uses HW1 ranking JSONL files for test/blind.
- The package uses `transformers.Trainer` instead of TRL `SFTTrainer` to avoid version-specific SFTTrainer argument changes.
