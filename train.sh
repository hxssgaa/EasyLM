export WANDB_API_KEY='9f081bf8abc9f49dffeb68c6cf978320514ab4b5'

# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

WANDB__SERVICE_WAIT=300 WANDB_MODE=offline python3 -m EasyLM.models.llama.llama_train \
    --total_steps=150000 \
    --mesh_dim='!1,32,1' \
    --log_freq=100 \
    --save_model_freq=1000 \
    --load_llama_config='7b' \
    --tokenizer.vocab_file='gs://hxtpu_bucket/llama2_tokenizer.model' \
    --load_checkpoint='params::gs://hxtpu_bucket/llama2_7b_easylm' \
    --llama.max_sequence_length=2048 \
    --train_dataset.text_processor.fields="text" \
    --train_dataset.type=huggingface \
    --train_dataset.huggingface_dataset.path='mc4' \
    --train_dataset.huggingface_dataset.name='en' \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.seq_length=2048 \
    --train_dataset.huggingface_dataset.batch_size=256 \
    --logger.output_dir='gs://hxtpu_bucket/llama2_mc4' \
    --logger.online=False \
    --logger.prefix='EasyLM' \
    --logger.project="my_llama2_7b" \
    --dtype=bf16 \
    --optimizer.adamw_optimizer.lr=1e-4 \
    --optimizer.adamw_optimizer.end_lr=5e-5 \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
    --optimizer.adamw_optimizer.lr_decay_steps=150000 \
    --optimizer.adamw_optimizer.bf16_momentum=True \
    --checkpointer.save_optimizer_state=False \
    --jax_distributed.initialize_jax_distributed=True


#--llama.remat_attention='checkpoint_dots' \