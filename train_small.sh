export WANDB_API_KEY='9f081bf8abc9f49dffeb68c6cf978320514ab4b5'

# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'
    # --load_checkpoint='params::gs://hxtpu_bucket/mistral_sea_7b_easylm' \
    # --tokenizer.vocab_file='gs://hxtpu_bucket/chinese_mistral_tokenizer.model' \
WANDB__SERVICE_WAIT=300 WANDB_MODE=offline python3 -m EasyLM.models.mistral.mistral_train \
    --total_steps=128000 \
    --mesh_dim='1,1,1'\
    --log_freq=1 \
    --save_model_freq=2048 \
    --load_mistral_config='debug' \
    --tokenizer.vocab_file='./inputs/mistral_tokenizer.model' \
    --mistral.max_sequence_length=8192 \
    --train_dataset.text_processor.fields="text" \
    --train_dataset.type=json \
    --train_dataset.json_dataset.path='/home/hxssgaa/Developer/seallama2/en_out.jsonl' \
    --train_dataset.json_dataset.batch_size=2 \
    --train_dataset.json_dataset.tokenizer_processes=16 \
    --train_dataset.json_dataset.seq_length=8192 \
    --logger.output_dir='./mistral_debug_output' \
    --logger.online=False \
    --logger.project="sea_mistral_7b" \
    --logger.experiment_id="mix_sea_mc" \
    --logger.prefix_to_id=True \
    --logger.prefix="EasyLM" \
    --dtype=bf16 \
    --optimizer.adamw_optimizer.lr=5e-4 \
    --optimizer.adamw_optimizer.end_lr=5e-5 \
    --optimizer.adamw_optimizer.b2=0.999 \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.lr_warmup_steps=2048 \
    --optimizer.adamw_optimizer.lr_decay_steps=150000 \
    --optimizer.adamw_optimizer.bf16_momentum=True \
    --checkpointer.save_optimizer_state=False

#    --load_checkpoint='params::gs://hxtpu_bucket/llama2_7b_easylm' \
# --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
#--llama.remat_attention='checkpoint_dots' \
#256 batch size
