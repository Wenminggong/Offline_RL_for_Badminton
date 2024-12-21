#!/bin/bash

# domain metrics evaluation for CQL
policy_seeds=(0 1024 2024)

mkdir -p domain_metrics_logs

for policy_seed in "${policy_seeds[@]}"; do
    echo "new cql domain metrics instance start."
    python domain_metrics_tactics_main.py \
        --save_path "domain_metrics_save_final" \
        --policy_type "cql" \
        --policy_path "policy_models_save_final" \
        --policy_batch_size 256 \
        --cql_target_action_gap 5.0 \
        --cql_tune_init_log_alpha -2.0 \
        --policy_embedding_dim 32 \
        --policy_hidden_dims 256 \
        --policy_n_hidden_layers 3 \
        --policy_seed $policy_seed \
        > domain_metrics_logs/domain_metrics_cql_s${policy_seed}.log 2>&1 &
    wait
done


echo "cql evaluation done!"