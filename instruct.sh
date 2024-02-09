export WANDB_API_KEY='9f081bf8abc9f49dffeb68c6cf978320514ab4b5'

# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'
    # --load_checkpoint='params::gs://hxtpu_bucket/mistral_sea_7b_easylm' \
    # --tokenizer.vocab_file='gs://hxtpu_bucket/chinese_mistral_tokenizer.model' \
WANDB__SERVICE_WAIT=300 WANDB_MODE=offline python3 -m EasyLM.models.mistral.mistral_train \
    --total_steps=8192 \
    --mesh_dim='1,16,-1'\
    --log_freq=32 \
    --eval_steps=16 \
    --save_model_freq=64 \
    --save_milestone_freq=0 \
    --save_best=True \
    --best_metric='eval_accuracy' \
    --load_mistral_config='7b_original_lora' \
    --tokenizer.vocab_file='gs://hxtpu_bucket/mistral_tokenizer.model' \
    --load_checkpoint='params::gs://hxtpu_bucket/mistral_easylm' \
    --mistral.max_sequence_length=8192 \
    --train_dataset.text_processor.tag="language" \
    --train_dataset.type=json \
    --train_dataset.text_processor_class='InstructTextProcessor' \
    --train_dataset.json_dataset.path='gs://hxtpu_bucket/star_instruction.jsonl' \
    --train_dataset.json_dataset.batch_size=32 \
    --train_dataset.json_dataset.enable_padding=True \
    --train_dataset.json_dataset.tokenizer_processes=16 \
    --train_dataset.json_dataset.seq_length=8192 \
    --eval_dataset.text_processor.tag="category" \
    --eval_dataset.type=json \
    --eval_dataset.text_processor_class='InstructSingleChoiceTextProcessor' \
    --eval_dataset.json_dataset.path='gs://hxtpu_bucket/sgeval_lite.jsonl' \
    --eval_dataset.json_dataset.batch_size=32 \
    --eval_dataset.json_dataset.enable_padding=True \
    --eval_dataset.json_dataset.tokenizer_processes=16 \
    --eval_dataset.json_dataset.seq_length=8192 \
    --logger.output_dir='gs://hxtpu_bucket/sea_mistral_7b_star_inst_outputs/' \
    --logger.online=True \
    --logger.project="sea_mistral_7b" \
    --logger.experiment_id="mix_sea_mc" \
    --logger.prefix_to_id=True \
    --logger.prefix="EasyLM-$1" \
    --dtype=bf16 \
    --optimizer.adamw_optimizer.lr=2e-4 \
    --optimizer.adamw_optimizer.end_lr=2e-5 \
    --optimizer.adamw_optimizer.enable_lora=True \
    --optimizer.adamw_optimizer.b2=0.999 \
    --optimizer.accumulate_gradient_steps=2 \
    --optimizer.adamw_optimizer.lr_warmup_steps=256 \
    --optimizer.adamw_optimizer.lr_decay_steps=8192 \
    --optimizer.adamw_optimizer.bf16_momentum=True \
    --checkpointer.save_optimizer_state=False \
    --jax_distributed.initialize_jax_distributed=True

# Alternative model:
# Continue pretrained SEA Mistral vocab file: gs://hxtpu_bucket/chinese_mistral_tokenizer.model
# Continue pretrained SEA Mistral model: trainstate_params::gs://hxtpu_bucket/sea_mistral_7b_outputs/mix_sea_mc/streaming_train_state
# Original Mistral vocab file gs://hxtpu_bucket/mistral_tokenizer.model
# Original Mistral model: gs://hxtpu_bucket/mistral_easylm

# Enable LoRA:
# `optimizer.adamw_optimizer.enable_lora=True` and `load_mistral_config='7b_lora'`
# Disable LoRA:
# `optimizer.adamw_optimizer.enable_lora=False` and `load_mistral_config='7b'`

# Alternative dataset for train_dataset:
# English-only high quality dataset: gs://hxtpu_bucket/openhermes2_5.jsonl
# English-Chinese dataset: gs://hxtpu_bucket/star_instruction.jsonl

# Get status of TPU:
# gcloud alpha compute tpus tpu-vm list (STATUS=ready means the TPU is on) (STATUS=PREEMPTED means the TPU is preempted)

# Delete TPU: (Only if the TPU is preempted)
# bash remove_tpu.sh (Wait until the commandline to finish)

# Create TPU: (Only if there is no TPU)
# bash create_tpu.sh (use gcloud alpha compute tpus tpu-vm list to check TPU status after a couple of minutes, wait until STATUS=READY)

# Setup environments: (Only if the TPU is newly created)
# bash easylm_setup.sh (wait the command to finish)

# Remove existing jobs: (To make sure the jobs are deleted)
# bash stop_easylm.sh (stop all jobs, wait until commands finish)

# Commit training job: (Make sure the jobs are deleted before commiting training job)
# bash instruct_easylm.sh abc (To start training jobs for given tags)

#    --load_checkpoint='params::gs://hxtpu_bucket/llama2_7b_easylm' \
# --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
#--llama.remat_attention='checkpoint_dots' \
#256 batch size