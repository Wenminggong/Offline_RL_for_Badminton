#!/bin/bash

# preference-based reward learning train
pref_model=$1
loss_type=$2
action_pref_type=$3
if [ "${loss_type}" = 0 ]; then
    action_pref_factors=(1.0)
else
    action_pref_factors=(0.5 1.0 3.0)
fi
ensemble_sizes=(1 5)
batch_sizes=(256 512)
learning_rates=(1e-4 1e-5)
hidden_dims=(256 512)
n_layers=(3 5)

mkdir -p reward_learn_logs

for action_pref_factor in "${action_pref_factors[@]}"; do
    for ensemble_size in "${ensemble_sizes[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for learning_rate in "${learning_rates[@]}"; do
                for hidden_dim in "${hidden_dims[@]}"; do
                    for n_layer in "${n_layers[@]}"; do
                        echo "new reward model training instance start."
                        python reward_learn_main.py \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --hidden_dim $hidden_dim \
                            --n_layer $n_layer \
                            --pref_model $pref_model \
                            --loss_type $loss_type \
                            --ensemble_size $ensemble_size \
                            --action_pref_type $action_pref_type \
                            --action_pref_factor $action_pref_factor \
                            > reward_learn_logs/reward_pm${pref_model}_lt${loss_type}_at${action_pref_type}_af${action_pref_factor}_es${ensemble_size}_b${batch_size}_l${learning_rate}_h${hidden_dim}_n${n_layer}.log 2>&1 &
                        wait
                    done
                done
            done
        done
    done
done

echo "reward model training done!"