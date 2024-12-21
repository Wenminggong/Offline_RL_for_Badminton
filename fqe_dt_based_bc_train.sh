#!/bin/bash

# OPE hyper-parameter and random seed for DT-based BC model
batch_sizes=(128)
hidden_sizes=(256 512)
embed_sizes=(32 64)
n_layers=(3 6)
n_heads=(2 8)
seeds=(2024)
fqe_n_hidden_layers=(3)
fqe_embed_sizes=(32)

mkdir -p ope_logs

for batch_size in "${batch_sizes[@]}"; do
    for hidden_size in "${hidden_sizes[@]}"; do
        for embed_size in "${embed_sizes[@]}"; do
            for n_layer in "${n_layers[@]}"; do
                for n_head in "${n_heads[@]}"; do
                    for policy_seed in "${seeds[@]}"; do
                        for fqe_n_layer in "${fqe_n_hidden_layers[@]}"; do
                            for fqe_embed_size in "${fqe_embed_sizes[@]}"; do
                                echo "new ope-dt-bc training instance start."
                                python ope_tactics_main.py \
                                    --seed 2024 \
                                    --batch_size 256 \
                                    --qf_lr 1e-6 \
                                    --q_n_hidden_layers $fqe_n_layer \
                                    --q_hidden_dims 256 \
                                    --embedding_dim $fqe_embed_size \
                                    --max_timesteps 400000 \
                                    --policy_type "bc" \
                                    --policy_seed $policy_seed \
                                    --policy_batch_size $batch_size \
                                    --policy_hidden_dims $hidden_size \
                                    --policy_embedding_dim $embed_size \
                                    --n_layer $n_layer \
                                    --n_head $n_head \
                                    --use_win_return 1 \
                                    --policy_path "trained_models" \
                                    --checkpoints_path "evaluation_models_last" \
                                    > ope_logs/train_fqe_bc_fn${fqe_n_layer}_fe${fqe_embed_size}_b${batch_size}_h${hidden_size}_e${embed_size}_nl${n_layer}_nh${n_head}_s${policy_seed}_new.log 2>&1 &
                                wait
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "ope-dt-bc training done!"