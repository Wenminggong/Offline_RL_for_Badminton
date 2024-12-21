#!/bin/bash

# FQE train and evaluation for cql model
policy_seeds=(0 1024 2024)

mkdir -p fqe_logs

for policy_seed in "${policy_seeds[@]}"; do
    echo "new fqe-cql training instance start."
    python fqe_tactics_main.py \
        --seed 2024 \
        --batch_size 256 \
        --qf_lr 1e-6 \
        --q_n_hidden_layers 3 \
        --q_hidden_dims 256 \
        --embedding_dim 32 \
        --max_timesteps 400000 \
        --policy_type "cql" \
        --policy_batch_size 256 \
        --cql_target_action_gap 5.0 \
        --cql_tune_init_log_alpha -2.0 \
        --policy_embedding_dim 32 \
        --policy_hidden_dims 256 \
        --policy_n_hidden_layers 3 \
        --policy_seed $policy_seed \
        --policy_path "policy_models_save_final" \
        --checkpoints_path "fqe_models_save_final" \
        > fqe_logs/train_fqe_cql_s${policy_seed}.log 2>&1 &
    wait
done



echo "fqe-cql training done!"