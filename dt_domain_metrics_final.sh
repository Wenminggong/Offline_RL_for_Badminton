#!/bin/bash

# domain metrics evaluation for DT
policy_seeds=(0 1024 2024)

mkdir -p domain_metrics_logs

for policy_seed in "${policy_seeds[@]}"; do
    echo "new dt domain metrics instance start."
    python domain_metrics_tactics_main.py \
        --save_path "domain_metrics_save_final" \
        --policy_type "dt" \
        --policy_path "policy_models_save_final" \
        --policy_batch_size 128 \
        --policy_embedding_dim 64 \
        --policy_hidden_dims 512 \
        --n_layer 3 \
        --n_head 2 \
        --policy_seed $policy_seed \
        --use_win_return 1 \
        > domain_metrics_logs/domain_metrics_dt_s${policy_seed}.log 2>&1 &
    wait
done

echo "dt evaluation done!"