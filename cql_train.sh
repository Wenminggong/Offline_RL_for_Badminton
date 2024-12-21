#!/bin/bash

# CQL hyper-parameter and random seed
embed_sizes=(32)
n_layers=(3)
hidden_dims=(256 512)
cql_target_action_gaps=(5.0)
cql_tune_init_log_alphas=(-2.0 -1.0 0.0)
seeds=(2024)

mkdir -p cql_logs


for embed_size in "${embed_sizes[@]}"; do
    for n_layer in "${n_layers[@]}"; do
        for hidden_dim in "${hidden_dims[@]}"; do
            for cql_target_action_gap in "${cql_target_action_gaps[@]}"; do
                for cql_tune_init_log_alpha in "${cql_tune_init_log_alphas[@]}"; do
                    for seed in "${seeds[@]}"; do
                        echo "new cql training instance start."
                        python cql_tactics_main.py \
                            --batch_size 256 \
                            --cql_target_action_gap $cql_target_action_gap \
                            --cql_tune_init_log_alpha $cql_tune_init_log_alpha \
                            --embedding_dim $embed_size \
                            --q_hidden_dims $hidden_dim \
                            --q_n_hidden_layers $n_layer \
                            --policy_hidden_dims $hidden_dim \
                            --policy_n_hidden_layers $n_layer \
                            --seed $seed \
                            --max_timesteps 100000 \
                            --model_type "cql" \
                            --policy_lr 5e-7 \
                            --qf_lr 1e-6 \
                            > cql_logs/train_cql_h${hidden_dim}_nl${n_layer}_cg${cql_target_action_gap}_ca${cql_tune_init_log_alpha}_s${seed}.log 2>&1 &
                        wait
                    done
                done
            done
        done
    done
done


echo "cql training done!"