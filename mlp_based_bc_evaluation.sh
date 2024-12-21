#!/bin/bash

# MLP-based BC hyper-parameter and random seed
filter_types=("both" "win" "loss")
policy_batch_sizes=(256 512)
policy_hidden_sizes=(256 512)
policy_embed_sizes=(32 64)
policy_n_layers=(3 6)
policy_seeds=(2024)

mkdir -p evaluation_logs

for filter_type in "${filter_types[@]}"; do
    for policy_batch_size in "${policy_batch_sizes[@]}"; do
        for policy_hidden_size in "${policy_hidden_sizes[@]}"; do
            for policy_embed_size in "${policy_embed_sizes[@]}"; do
                for policy_n_layer in "${policy_n_layers[@]}"; do
                    for policy_seed in "${policy_seeds[@]}"; do
                        echo "new mlp-bc evaluation instance start."
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
                            --policy_type "mlp_bc" \
                            --policy_path "trained_models" \
                            --policy_batch_size $policy_batch_size \
                            --policy_embedding_dim $policy_embed_size \
                            --policy_hidden_dims $policy_hidden_size \
                            --policy_n_hidden_layers $policy_n_layer \
                            --policy_seed $policy_seed \
                            > evaluation_logs/eval_mlp_bc_reward_${filter_type}_b${policy_batch_size}_h${policy_hidden_size}_e${policy_embed_size}_nl${policy_n_layer}_s${policy_seed}.log 2>&1 &
                        wait
                    done
                done
            done
        done
    done
done

echo "mlp-bc evaluation done!"