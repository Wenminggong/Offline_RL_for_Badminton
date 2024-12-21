#!/bin/bash

# DT-based BC hyper-parameter and random seed
filter_types=("both" "win" "loss")
policy_batch_sizes=(128)
policy_hidden_sizes=(256 512)
policy_embed_sizes=(32 64)
n_layers=(3 6)
n_heads=(2 8)
policy_seeds=(2024)

mkdir -p evaluation_logs

for filter_type in "${filter_types[@]}"; do
    for policy_batch_size in "${policy_batch_sizes[@]}"; do
        for policy_hidden_size in "${policy_hidden_sizes[@]}"; do
            for policy_embed_size in "${policy_embed_sizes[@]}"; do
                for n_layer in "${n_layers[@]}"; do
                    for n_head in "${n_heads[@]}"; do
                        for policy_seed in "${policy_seeds[@]}"; do
                            echo "new dt-based bc evaluation instance start."
                            python tactics_model_evaluation.py \
                                --seed 2024 \
                                --filter_type $filter_type \
                                --evaluator_type "reward" \
                                --reward_pref_model 0 \
                                --reward_loss_type 1 \
                                --reward_ensemble_size 5 \
                                --reward_action_pref_type 1 \
                                --reward_learning_rate 1e-5 \
                                --reward_batch_size 256 \
                                --reward_action_pref_factor 1.0 \
                                --reward_n_layer 5 \
                                --reward_hidden_dim 256 \
                                --policy_type "bc" \
                                --policy_path "trained_models" \
                                --policy_seed $policy_seed \
                                --policy_batch_size $policy_batch_size \
                                --policy_embedding_dim $policy_embed_size \
                                --policy_hidden_dims $policy_hidden_size \
                                --n_layer $n_layer \
                                --n_head $n_head \
                                > evaluation_logs/eval_dt_bc_reward_${filter_type}_b${policy_batch_size}_h${policy_hidden_size}_e${policy_embed_size}_nl${n_layer}_nh${n_head}_s${policy_seed}.log 2>&1 &
                            wait
                        done
                    done
                done
            done
        done
    done
done

echo "dt-based bc evaluation done!"