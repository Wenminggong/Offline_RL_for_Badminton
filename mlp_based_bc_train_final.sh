#!/bin/bash

# MLP-BC training with optimal superparameters and 3 random seeds
seeds=(0 1024 2024)

mkdir -p mlp_based_bc_logs


for seed in "${seeds[@]}"; do
    echo "new mlp-bc training instance start."
    python cql_tactics_main.py \
        --batch_size 256 \
        --policy_hidden_dims 256 \
        --policy_n_hidden_layers 3 \
        --embedding_dim 32 \
        --seed $seed \
        --max_timesteps 100000 \
        --model_type "mlp_bc" \
        --eval_freq 500 \
        --policy_lr 1e-6 \
        --checkpoints_path "policy_models_save_final" \
        > mlp_based_bc_logs/train_mlp_bc_final_s${seed}.log 2>&1 &
    wait
done


echo "mlp-bc training done!"