#!/bin/bash

# preference-based reward learning train
batch_size=$1
hidden_dim=$2
n_layer=$3
pref_model=$4
use_no_end_rally_pref=$5
no_end_rally_pref_mode=$6
ensemble_size=$7
seeds=(0 1024 2024) 

mkdir -p reward_learn_logs

for seed in "${seeds[@]}"; do
    echo "new reward model training instance start."
    python reward_learn_main.py \
        --batch_size $batch_size \
        --model_save_path "reward_models_save_final" \
        --seed $seed \
        --hidden_dim $hidden_dim \
        --n_layer $n_layer \
        --pref_model $pref_model \
        --use_no_end_rally_pref $use_no_end_rally_pref \
        --no_end_rally_pref_mode $no_end_rally_pref_mode \
        --no_end_rally_pref_factor 0.5 \
        --loss_type 0 \
        --ensemble_size $ensemble_size \
        --action_pref_type 0 \
        > reward_learn_logs/reward_b${batch_size}_hd${hidden_dim}_nl${n_layer}_pm${pref_model}_unep${use_no_end_rally_pref}_nepm${no_end_rally_pref_mode}_es${ensemble_size}_seed${seed}.log 2>&1 &
    wait
done

echo "reward model training done!"