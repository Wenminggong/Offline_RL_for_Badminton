#!/bin/bash

# DT-based BC hyper-parameter and random seed
batch_sizes=(128)
hidden_sizes=(256 512)
embed_sizes=(32 64)
n_layers=(3 6)
n_heads=(2 8)
seeds=(2024)

mkdir -p dt_based_bc_logs

for batch_size in "${batch_sizes[@]}"; do
    for hidden_size in "${hidden_sizes[@]}"; do
        for embed_size in "${embed_sizes[@]}"; do
            for n_layer in "${n_layers[@]}"; do
                for n_head in "${n_heads[@]}"; do
                    for seed in "${seeds[@]}"; do
                        echo "new dtbc training instance start."
                        python dt_tactics_main.py \
                            --model_type "bc" \
                            --batch_size $batch_size \
                            --hidden_dim $hidden_size \
                            --embed_dim $embed_size \
                            --n_layer $n_layer \
                            --n_head $n_head \
                            --seed $seed \
                            --max_iters 200 \
                            --warmup_steps 10000 \
                            --num_steps_per_iter 500 \
                            --learning_rate 1e-6 \
                            --weight_decay 1e-3 \
                            --model_save_path "trained_models" \
                            > dt_based_bc_logs/train_dtbc_b${batch_size}_h${hidden_size}_e${embed_size}_nl${n_layer}_nh${n_head}_s${seed}.log 2>&1 &
                        wait
                    done
                done
            done
        done
    done
done

echo "dtbc training done!"