python convert_easylm_to_hf.py  \
       --load_checkpoint="trainstate_params::gs://hxtpu_bucket/sea_mistral_7b_star_inst_outputs/EasyLM-replay_lora_openhermes2_4--mix_sea_mc/streaming_train_state_15360"    \
       --output_dir="./sea_mistral_inst_7b"   \
       --tokenizer_path="./chinese_mistral_tokenizer.model" \
       --enable_lora=True \
       --model_size="7b"
