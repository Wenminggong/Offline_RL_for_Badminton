#!/bin/bash

# FQE train and evaluation for DT-based BC model
policy_seeds=(0 1024 2024)

mkdir -p fqe_logs

for policy_seed in "${policy_seeds[@]}"; do
    echo "new fqe-dt-bc training instance start."
    python fqe_tactics_main.py \
        --seed 2024 \
        --batch_size 256 \
        --qf_lr 1e-6 \
        --q_n_hidden_layers 3 \
        --q_hidden_dims 256 \
        --embedding_dim 32 \
        --max_timesteps 400000 \
        --policy_type "bc" \
        --policy_seed $policy_seed \
        --policy_batch_size 128 \
        --policy_hidden_dims 512 \
        --policy_embedding_dim 32 \
        --n_layer 3 \
        --n_head 2 \
        --use_win_return 1 \
        --policy_path "policy_models_save_final" \
        --checkpoints_path "fqe_models_save_final" \
        > fqe_logs/train_fqe_bc_s${policy_seed}.log 2>&1 &
    wait
done


echo "fqe-dt-bc training done!"