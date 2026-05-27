data_module = DataModule(seed=SEED)
ds = data_module.load(ds_url)
train_df, val_df = data_module.build_train_val_split()

# attach rankings so retrieved_chunks + ranking are available
retrieval_store = RetrievalStore(top_k=3)
train_df = retrieval_store.attach_rankings_from_jsonl(train_df, test_ranking_path)
val_df   = retrieval_store.attach_rankings_from_jsonl(val_df,   test_ranking_path)

prompt_builder = PromptBuilder()

# build SFT datasets
sft_builder = SFTDatasetBuilder(
    prompt_builder=prompt_builder,
    retrieval_store=retrieval_store,
    seed=SEED,
)

# need tokenizer for chat template formatting
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

train_sft = sft_builder.build_hf_dataset(
    train_df,
    tokenizer=tokenizer,
    variants=["oracle_first", "oracle_random", "no_answer"],
    no_answer_ratio=0.3,
)

val_sft = sft_builder.build_hf_dataset(
    val_df,
    tokenizer=tokenizer,
    variants=["oracle_first", "oracle_random", "no_answer"],
    no_answer_ratio=0.3,
)

print(f"Train: {len(train_sft)} | Val: {len(val_sft)}")
print(train_sft[0]["text"])

# finetune
finetuner = LLMFineTuner(
    model_name="Qwen/Qwen3-0.6B",
    output_dir=os.path.join(output_dir, "sft-qwen3-0.6b"),
    lora_r=16,
    lora_alpha=32,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
)

model, tokenizer, trainer = finetuner.finetune(
    train_dataset=train_sft,
    val_dataset=val_sft,
)