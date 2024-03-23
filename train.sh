export WANDB_API_KEY='9f081bf8abc9f49dffeb68c6cf978320514ab4b5'

# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'
    # --load_checkpoint='params::gs://hxtpu_bucket/mistral_sea_7b_easylm' \
    # --tokenizer.vocab_file='gs://hxtpu_bucket/chinese_mistral_tokenizer.model' \

#     --load_checkpoint='trainstate_params::gs://hxtpu_bucket/sea_mistral_7b_outputs/mix_sea_mc/streaming_train_state' \
# --save_model_freq=512
WANDB__SERVICE_WAIT=300 WANDB_MODE=offline python3 -m EasyLM.models.mistral.mistral_train \
    --total_steps=128000 \
    --mesh_dim='1,1,4,16'\
    --dtype='bf16' \
    --log_freq=512 \
    --eval_steps=0 \
    --save_model_freq=0 \
    --save_milestone_freq=10240 \
    --save_best=True \
    --best_metric='eval_accuracy' \
    --load_mistral_config='7b' \
    --update_mistral_config="dict(max_sequence_length=262144,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=1024,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=512,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=False,attention_type='ring_blockwise',param_scan_axis=0,mesh_dim='1,1,4,16')" \
    --tokenizer.vocab_file='gs://hxtpu_bucket/chinese_mistral_tokenizer.model' \
    --train_dataset.text_processor.fields="text" \
    --train_dataset.text_processor.tag="language" \
    --train_dataset.type=json \
    --train_dataset.json_dataset.path='gs://hxtpu_bucket/v3_sg_openhermes.slimpajama.train.instruct.jsonl' \
    --train_dataset.json_dataset.batch_size=1 \
    --train_dataset.json_dataset.tokenizer_processes=16 \
    --train_dataset.json_dataset.seq_length=262144 \
    --eval_dataset.text_processor.tag="category" \
    --eval_dataset.type=json \
    --eval_dataset.text_processor_class='InstructSingleChoiceTextProcessor' \
    --eval_dataset.json_dataset.path='gs://hxtpu_bucket/sgeval_lite.jsonl' \
    --eval_dataset.json_dataset.batch_size=1 \
    --eval_dataset.json_dataset.enable_padding=True \
    --eval_dataset.json_dataset.tokenizer_processes=16 \
    --eval_dataset.json_dataset.seq_length=262144 \
    --logger.output_dir='gs://hxtpu_bucket/regional_sea_mistral_7b_long_context' \
    --logger.online=True \
    --logger.project="regional_sea_mistral_7b" \
    --logger.experiment_id="long_context" \
    --logger.prefix_to_id=True \
    --logger.prefix="EasyLM-$1" \
    --optimizer.adamw_optimizer.lr=5e-5 \
    --optimizer.adamw_optimizer.end_lr=1e-5 \
    --optimizer.adamw_optimizer.b2=0.999 \
    --optimizer.accumulate_gradient_steps=16 \
    --optimizer.adamw_optimizer.lr_warmup_steps=2048 \
    --optimizer.adamw_optimizer.lr_decay_steps=150000 \
    --optimizer.adamw_optimizer.bf16_momentum=True \
    --checkpointer.save_optimizer_state=True \
    --jax_distributed.initialize_jax_distributed=True

#    --load_checkpoint='params::gs://hxtpu_bucket/llama2_7b_easylm' \
# --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
#--llama.remat_attention='checkpoint_dots' \
#256 batch size
    # --eval_dataset.text_processor.tag="category" \
    # --eval_dataset.type=json \
    # --eval_dataset.text_processor_class='InstructSingleChoiceTextProcessor' \
    # --eval_dataset.json_dataset.path='gs://hxtpu_bucket/sgeval_lite.jsonl' \
    # --eval_dataset.json_dataset.batch_size=128 \
    # --eval_dataset.json_dataset.enable_padding=True \
    # --eval_dataset.json_dataset.tokenizer_processes=16 \
    # --eval_dataset.json_dataset.seq_length=8192 \