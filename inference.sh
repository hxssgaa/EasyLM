python -m EasyLM.models.mistral.mistral_serve \
    --load_mistral_config='7b' \
    --load_checkpoint="trainstate_params::gs://hxtpu_bucket/sea_mistral_7b_outputs/mix_sea_mc/streaming_train_state" \
    --tokenizer.vocab_file='gs://hxtpu_bucket/chinese_mistral_tokenizer.model' \
    --mesh_dim='1,-1,1' \
    --dtype='bf16' \
    --input_length=1024 \
    --seq_length=2048 \
    --lm_server.batch_size=4 \
    --lm_server.port=35009 \
    --lm_server.pre_compile='all'