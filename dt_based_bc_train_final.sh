#!/bin/bash

# DT-based BC training with optimal superparameters and 3 random seeds
seeds=(0 1024 2024)

mkdir -p dt_based_bc_logs

for seed in "${seeds[@]}"; do
    echo "new dtbc training instance start."
    python dt_tactics_main.py \
        --model_type "bc" \
        --batch_size 128 \
        --hidden_dim 512 \
        --embed_dim 32 \
        --n_layer 3 \
        --n_head 2 \
        --seed $seed \
        --max_iters 200 \
        --warmup_steps 10000 \
        --num_steps_per_iter 500 \
        --learning_rate 1e-6 \
        --weight_decay 1e-3 \
        --model_save_path "policy_models_save_final" \
        > dt_based_bc_logs/train_dtbc_final_s${seed}.log 2>&1 &
    wait
done

echo "dtbc training done!"