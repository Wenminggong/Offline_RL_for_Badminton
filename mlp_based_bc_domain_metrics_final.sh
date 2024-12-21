#!/bin/bash

# domain metrics evaluation for mlp-bc
policy_seeds=(0 1024 2024)

mkdir -p domain_metrics_logs

for policy_seed in "${policy_seeds[@]}"; do
    echo "new mlp-bc domain metrics instance start."
    python domain_metrics_tactics_main.py \
        --save_path "domain_metrics_save_final" \
        --policy_type "mlp_bc" \
        --policy_path "policy_models_save_final" \
        --policy_batch_size 256 \
        --policy_embedding_dim 32 \
        --policy_hidden_dims 256 \
        --policy_n_hidden_layers 3 \
        --policy_seed $policy_seed \
        > domain_metrics_logs/domain_metrics_mlp_bc_s${policy_seed}.log 2>&1 &
    wait
done

echo "mlp-bc evaluation done!"