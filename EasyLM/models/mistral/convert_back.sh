python convert_easylm_to_hf.py  \
       --load_checkpoint="trainstate_params::gs://hxtpu_bucket/sea_mistral_7b_outputs/mix_sea_mc/streaming_train_state"    \
       --output_dir="./sea_mistral_7b"   \
       --tokenizer_path="gs://hxtpu_bucket/chinese_mistral_tokenizer.model"
