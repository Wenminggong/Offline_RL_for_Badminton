#!/bin/bash

# evaluating the average reward of the behavior policy used to generate the dataset by using reward model.
filter_types=("loss" "win" "both")

mkdir -p evaluation_logs

for filter_type in "${filter_types[@]}"; do
    echo "new original evaluation instance start."
    python tactics_model_evaluation.py \
        --seed 2024 \
        --save_path "evaluation_results_final_without_terminal" \
        --filter_type $filter_type \
        --evaluator_type "reward" \
        --reward_pref_model 0 \
        --reward_use_no_end_rally_pref 0 \
        --reward_no_end_rally_pref_mode 0 \
        --reward_no_end_rally_pref_factor 0.5 \
        --reward_loss_type 0 \
        --reward_ensemble_size 5 \
        --reward_action_pref_type 0 \
        --reward_action_pref_factor 1.0 \
        --reward_learning_rate 1e-4 \
        --reward_batch_size 512 \
        --reward_n_layer 5 \
        --reward_hidden_dim 512 \
        --reward_model_dir "reward_models_save_final" \
        --policy_type "original" \
        > evaluation_logs/eval_original_reward_${filter_type}.log 2>&1 &
    wait
done

echo "original evaluation done!"