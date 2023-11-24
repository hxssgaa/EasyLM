export WANDB_API_KEY='9f081bf8abc9f49dffeb68c6cf978320514ab4b5'

# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

WANDB__SERVICE_WAIT=300 WANDB_MODE=online python3 -m EasyLM.models.mistral.mistral_train \
    --total_steps=150000 \
    --mesh_dim='1,16,-1' \
    --log_freq=200 \
    --save_model_freq=1000 \
    --load_llama_config='7b' \
    --tokenizer.vocab_file='gs://hxtpu_bucket/mistral_tokenizer.model' \
    --load_checkpoint='params::gs://hxtpu_bucket/mistral_easylm' \
    --llama.max_sequence_length=4096 \
    --train_dataset.text_processor.fields="raw_content" \
    --train_dataset.type=huggingface \
    --train_dataset.huggingface_dataset.path='togethercomputer/RedPajama-Data-V2' \
    --train_dataset.huggingface_dataset.name='default' \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.seq_length=4096 \
    --train_dataset.huggingface_dataset.batch_size=256 \
    --train_dataset.huggingface_dataset.dataset_sample_prob='0.5,0.5' \
    --logger.output_dir='gs://hxtpu_bucket/mistral_outputs' \
    --logger.online=True \
    --logger.prefix='EasyLM' \
    --logger.project="my_mistral_7b" \
    --dtype=bf16 \
    --optimizer.adamw_optimizer.lr=5e-5 \
    --optimizer.adamw_optimizer.end_lr=1e-5 \
    --optimizer.accumulate_gradient_steps=8 \
    --optimizer.adamw_optimizer.lr_warmup_steps=200 \
    --optimizer.adamw_optimizer.lr_decay_steps=150000 \
    --optimizer.adamw_optimizer.bf16_momentum=True \
    --checkpointer.save_optimizer_state=False \
    --jax_distributed.initialize_jax_distributed=True

#    --load_checkpoint='params::gs://hxtpu_bucket/llama2_7b_easylm' \
# --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
#--llama.remat_attention='checkpoint_dots' \
#256 batch size