# run on 8xH20
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

TRAIN_DATA="$HOME/data/searchR1_processed_w_selective_plan/train.parquet"
VAL_DATA="$HOME/data/searchR1_processed_w_selective_plan/test.parquet"

# TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"
TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config_with_thinking_wo_status.yaml"

clip_ratio_low=0.2
clip_ratio_high=0.2
kl_cov_ratio=0.002
ppo_kl_coef=1
loss_mode="kl_cov"
loss_agg_mode="token-mean"

train_prompt_bsz=256
val_prompt_bsz=128


python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_prompt_bsz} \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Seungyoun/Qwen3-1.7B-Non-Thinking \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio=${kl_cov_ratio} \
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=${ppo_kl_coef} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.attention_backend=flashinfer \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='search_r1_like_async_rl' \
    trainer.experiment_name='qwen3-1.7b-plan-search-wo-status' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=60 \
    trainer.resume_mode=auto \
    reward_model.reward_manager=dapo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.total_epochs=1 $@

