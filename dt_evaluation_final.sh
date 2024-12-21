#!/bin/bash

# evaluating the average reward of DT used to generate the dataset by using reward model.
filter_types=("both" "win" "loss")
policy_seeds=(0 1024 2024)

mkdir -p evaluation_logs

for filter_type in "${filter_types[@]}"; do
    for policy_seed in "${policy_seeds[@]}"; do
        echo "new dt evaluation instance start."
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
            --policy_type "dt" \
            --policy_path "policy_models_save_final" \
            --policy_batch_size 128 \
            --policy_embedding_dim 64 \
            --policy_hidden_dims 512 \
            --n_layer 3 \
            --n_head 2 \
            --policy_seed $policy_seed \
            --use_win_return 0 \
            > evaluation_logs/eval_dt_reward_${filter_type}_s${policy_seed}.log 2>&1 &
        wait
    done
done

echo "dt evaluation done!"