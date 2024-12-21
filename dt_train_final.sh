#!/bin/bash

# DT training with optimal superparameters and 3 random seeds
seeds=(0 1024 2024)

mkdir -p dt_logs

for seed in "${seeds[@]}"; do
    echo "new dt training instance start."
    python dt_tactics_main.py \
        --batch_size 128 \
        --model_type "dt" \
        --hidden_dim 512 \
        --embed_dim 64 \
        --n_layer 3 \
        --n_head 2 \
        --seed $seed \
        --max_iters 200 \
        --warmup_steps 10000 \
        --num_steps_per_iter 500 \
        --learning_rate 1e-6 \
        --weight_decay 1e-3 \
        --model_save_path "policy_models_save_final" \
        > dt_logs/train_dt_final_s${seed}.log 2>&1 &
    wait
done

echo "dt training done!"