from mnlp_ananas2 import SFTDatasetBuilder, DataModule

data_module = DataModule(dev_size=0.2)
ds = data_module.load(ds_url)

train_df, val_df = data_module.build_train_val_split()
test_df = data_module.to_dataframe("test")
blind_df = data_module.to_dataframe("blind")

sft_builder = SFTDatasetBuilder(
    prompt_builder=prompt_builder,
    top_k=3,
    seed=SEED,
)

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