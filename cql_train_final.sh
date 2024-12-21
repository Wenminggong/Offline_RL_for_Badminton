#!/bin/bash

# CQL training with optimal superparameters and 3 random seeds
seeds=(0 1024 2024)

mkdir -p cql_logs

for seed in "${seeds[@]}"; do
    echo "new cql training instance start."
    python cql_tactics_main.py \
        --seed $seed \
        --checkpoints_path "policy_models_save_final" \
        --model_type "cql" \
        > cql_logs/train_cql_final_s${seed}.log 2>&1 &
    wait
done


echo "cql training done!"