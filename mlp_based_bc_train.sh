#!/bin/bash

# MLP-BC hyper-parameter and random seed
batch_sizes=(256 512)
hidden_sizes=(256 512)
embed_sizes=(32 64)
n_layers=(3 6)
seeds=(2024)

mkdir -p mlp_based_bc_logs

for batch_size in "${batch_sizes[@]}"; do
    for hidden_size in "${hidden_sizes[@]}"; do
        for embed_size in "${embed_sizes[@]}"; do
            for n_layer in "${n_layers[@]}"; do
                for seed in "${seeds[@]}"; do
                    echo "new mlp-bc training instance start."
                    python cql_tactics_main.py \
                        --batch_size $batch_size \
                        --embedding_dim $embed_size \
                        --policy_hidden_dims $hidden_size \
                        --policy_n_hidden_layers $n_layer \
                        --seed $seed \
                        --max_timesteps 100000 \
                        --model_type "mlp_bc" \
                        --eval_freq 500 \
                        --policy_lr 1e-6 \
                        > mlp_based_bc_logs/train_mlp_bc_b${batch_size}_h${hidden_size}_e${embed_size}_nl${n_layer}_s${seed}.log 2>&1 &
                    wait
                done
            done
        done
    done
done

echo "mlp-bc training done!"