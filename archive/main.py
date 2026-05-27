"""# Experiments

### Get drive + files
"""

from google.colab import drive
drive.mount('/content/drive')

data_module = DataModule(dev_size=0.2)
ds = data_module.load(ds_url)

train_df, val_df = data_module.build_train_val_split()
test_df = data_module.to_dataframe("test")
blind_df = data_module.to_dataframe("blind")

retrieval_store = RetrievalStore()
prompt_builder = PromptBuilder()

baseline_test_df = retrieval_store.attach_rankings_from_jsonl(
    test_df,
    test_ranking_path,
)

baseline_blind_df = retrieval_store.attach_rankings_from_jsonl(
    blind_df,
    blind_ranking_path,
)

runner = ExperimentRunner(
    prompt_builder,
    retrieval_store,
)

"""## [B1]: Implement at least two small LMs (<= 3B parameters) zero-shot in (a) baseline, (b) RAG and (c) Oracle settings

### Qwen3-0.6B
"""

qwen3_06B_model = QALLM(model_name="Qwen/Qwen3-0.6B")

"""Baseline setting"""

results = runner.retrieve_answers(
    test_df,
    qwen3_06B_model,
    setting="baseline",
)

display_results_table(results, n=50)

export_jsonl(
    results,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-all-test-Qwen3-0.6B-b1-baseline.jsonl",
)

AnswerEvaluator.evaluate_dataset(results)

results = runner.retrieve_answers(
    blind_df,
    qwen3_06B_model,
    setting="baseline",
)

export_jsonl(
    results,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-all-blind-Qwen3-0.6B-b1-baseline.jsonl",
)

"""RAG setting"""

qwen3_06B_rag_base_test = runner.retrieve_answers(
    df=baseline_test_df,
    model=qwen3_06B_model,
    setting="rag",
)

export_jsonl(
    qwen3_06B_rag_base_test,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-all-test-Qwen3-0.6B-b1-rag.jsonl",
)

results = AnswerEvaluator.evaluate_dataset(qwen3_06B_rag_base_test)

results

qwen3_06B_rag_base_blind = runner.retrieve_answers(
    df=baseline_blind_df,
    model=qwen3_06B_model,
    setting="rag",
)

export_jsonl(
    qwen3_06B_rag_base_blind,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-all-blind-Qwen3-0.6B-b1-rag.jsonl",
)

"""Oracle setting"""

qwen3_06B_orac_base_test = runner.retrieve_answers(
    df=baseline_test_df,
    model=qwen3_06B_model,
    setting="oracle",
)

export_jsonl(
    qwen3_06B_orac_base_test,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-all-test-Qwen3-0.6B-b1-oracle.jsonl",
)

display_results_table(qwen3_06B_orac_base_test, n=5)

AnswerEvaluator.evaluate_dataset(qwen3_06B_orac_base_test)

"""### SmolLM2-1.7B"""

smollm2_1_7B_model = SmolLM()

"""Baseline setting"""

smollm2_1_7B_base_test = runner.retrieve_answers(
    test_df,
    smollm2_1_7B_model,
    setting="baseline",
)

display_results_table(smollm2_1_7B_base_test, n=5)

export_jsonl(
    smollm2_1_7B_base_test,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-all-test-smollm2-7B-b1-baseline.jsonl",
)

AnswerEvaluator.evaluate_dataset(smollm2_1_7B_base_test)

smollm2_1_7B_base_blind = runner.retrieve_answers(
    blind_df,
    smollm2_1_7B_model,
    setting="baseline",
)

export_jsonl(
    smollm2_1_7B_base_blind,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-all-blind-smollm2-7B-b1-baseline.jsonl",
)

"""RAG setting"""

smollm2_1_7B_rag_base_test = runner.retrieve_answers(
    df=baseline_test_df,
    model=smollm2_1_7B_model,
    setting="rag",
)

export_jsonl(
    smollm2_1_7B_rag_base_test,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-all-test-SmolLM2-1.7B-b1-rag.jsonl",
)

AnswerEvaluator.evaluate_dataset(smollm2_1_7B_rag_base_test)

smollm2_1_7B_rag_base_blind = runner.retrieve_answers(
    df=baseline_blind_df,
    model=smollm2_1_7B_model,
    setting="rag",
)

export_jsonl(
    smollm2_1_7B_rag_base_blind,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-all-blind-SmolLM2-1.7B-b1-rag.jsonl",
)

"""Oracle setting"""

smollm2_1_7B_orac_base_test = runner.retrieve_answers(
    df=baseline_test_df,
    model=smollm2_1_7B_model,
    setting="oracle",
)

display_results_table(smollm2_1_7B_orac_base_test, n=5)

export_jsonl(
    smollm2_1_7B_orac_base_test,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-all-test-SmolLM2-1.7B-b1-oracle.jsonl",
)

AnswerEvaluator.evaluate_dataset(smollm2_1_7B_orac_base_test)

"""## [B2]: Evaluation Framework"""

generated_df = retrieval_store.load_generated_answers_jsonl(generated_path)

judge_df = retrieval_store.enrich_generated_answers(
    generated_df=generated_df,
    reference_df=test_df,
)

judge_subset_df = judge_df.sample(
    n=200,
    random_state=SEED,
).reset_index(drop=True)

judge_subset_df[[
    "query_id",
    "query",
    "short_answer",
    "generated_answer",
    "retrieved_chunks",
]].head()

# judge_model = LLMJudge(
#     model_name="prometheus-eval/prometheus-7b-v2.0",
#     max_new_tokens=16,
# )

# Or if it's too heavy:
#
judge_model = LLMJudge(
    model_name="Qwen/Qwen3-4B",
    max_new_tokens=8,
    supports_thinking=True
)

judge_subset_scored = judge_model.judge_dataframe(
    judge_subset_df,
    batch_size=4, # Change for models other than prometheus
)

export_jsonl(
    judge_subset_scored,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-judge-subset-Qwen3-4B-v1.jsonl",
    include_eval=True,
)

judge_subset_scored

manual_df = make_manual_annotation_csv(
    judge_jsonl_path=llm_judge_output_path,
    reference_df=test_df,
    output_csv_path=f"/content/drive/MyDrive/{output_dir}/outputs/manual_annotation_template.csv",
)

"""## [A1]: Use different prompting strategies

Chain of thoughts
"""

qwen3_06B_th_model = QALLM(model_name="Qwen/Qwen3-0.6B")

qwen3_06B_rag_th_test = runner.retrieve_answers(
    df=baseline_test_df,
    model=qwen3_06B_th_model,
    setting="rag",
)

export_jsonl(
    qwen3_06B_rag_th_test,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-all-test-Qwen3-0.6B-th-rag.jsonl",
)

results = AnswerEvaluator.evaluate_dataset(qwen3_06B_rag_base_test)

qwen3_06B_rag_th_blind = runner.retrieve_answers(
    df=baseline_blind_df,
    model=qwen3_06B_th_model,
    setting="rag",
)

export_jsonl(
    qwen3_06B_rag_th_blind,
    f"/content/drive/MyDrive/{output_dir}/outputs/doubleN-all-blind-Qwen3-0.6B-th-rag.jsonl",
)

"""## [A3]: Fine-tune a small LM for RAG"""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_dir = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model.config.use_cache = False
model.config.pretraining_tp = 1

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# LoRA config
peft_config = LoraConfig(
    lora_alpha=16,                           # Scaling factor for LoRA
    lora_dropout=0.05,                       # Add slight dropout for regularization
    r=64,                                    # Rank of the LoRA update matrices
    bias="none",                             # No bias reparameterization
    task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, peft_config)

# Training Arguments
training_arguments = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    logging_steps=0.2,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="none"
)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    peft_config=peft_config,
    data_collator=data_collator,
)

gc.collect()
torch.cuda.empty_cache()
model.config.use_cache = False
trainer.train()