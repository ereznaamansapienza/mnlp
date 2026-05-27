sft_builder = SFTDatasetBuilder(
    prompt_builder=prompt_builder,
    top_k=3,
    seed=SEED,
)

train_sft = sft_builder.build_hf_dataset(
    train_df,       # just needs candidate_chunks, answer_pos, query, short_answer
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