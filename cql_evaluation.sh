#!/bin/bash

# CQL hyper-parameter and random seed
filter_types=("loss")
policy_embed_sizes=(32)
policy_n_hidden_layers=(3)
policy_hidden_dims=(256 512)
cql_target_action_gaps=(5.0)
cql_tune_init_log_alphas=(-2.0 -1.0 0.0)
policy_seeds=(2024)

mkdir -p evaluation_logs

for filter_type in "${filter_types[@]}"; do
    for policy_embed_size in "${policy_embed_sizes[@]}"; do
        for policy_n_hidden_layer in "${policy_n_hidden_layers[@]}"; do
            for policy_hidden_dim in "${policy_hidden_dims[@]}"; do
                for cql_target_action_gap in "${cql_target_action_gaps[@]}"; do
                    for cql_tune_init_log_alpha in "${cql_tune_init_log_alphas[@]}"; do
                        for policy_seed in "${policy_seeds[@]}"; do
                            echo "new cql evaluation instance start."
                            python tactics_model_evaluation.py \
                                --seed 2024 \
                                --filter_type $filter_type \
                                --evaluator_type "reward" \
                                --reward_pref_model 0 \
                                --reward_loss_type 1 \
                                --reward_ensemble_size 5 \
                                --reward_action_pref_type 0 \
                                --reward_learning_rate 1e-4 \
                                --reward_batch_size 256 \
                                --reward_action_pref_factor 1.0 \
                                --reward_n_layer 5 \
                                --reward_hidden_dim 512 \
                                --policy_type "cql" \
                                --policy_path "trained_models" \
                                --policy_batch_size 256 \
                                --cql_target_action_gap $cql_target_action_gap \
                                --cql_tune_init_log_alpha $cql_tune_init_log_alpha \
                                --policy_embedding_dim $policy_embed_size \
                                --policy_hidden_dims $policy_hidden_dim \
                                --policy_n_hidden_layers $policy_n_hidden_layer \
                                --policy_seed $policy_seed \
                                > evaluation_logs/eval_cql_reward_${filter_type}_h${policy_hidden_dim}_nl${policy_n_hidden_layer}_cg${cql_target_action_gap}_ca${cql_tune_init_log_alpha}_s${policy_seed}.log 2>&1 &
                            wait
                        done
                    done
                done
            done
        done
    done
done


echo "cql evaluation done!"