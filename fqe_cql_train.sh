#!/bin/bash

# ope hyper-parameter and random seed for cql model
embed_sizes=(32)
n_layers=(3)
hidden_dims=(256 512)
cql_target_action_gaps=(5.0)
cql_tune_init_log_alphas=(-2.0 -1.0 0.0)
seeds=(2024)
fqe_n_hidden_layers=(3)
fqe_embed_sizes=(32)

mkdir -p ope_logs


for embed_size in "${embed_sizes[@]}"; do
    for n_layer in "${n_layers[@]}"; do
        for hidden_dim in "${hidden_dims[@]}"; do
            for cql_target_action_gap in "${cql_target_action_gaps[@]}"; do
                for cql_tune_init_log_alpha in "${cql_tune_init_log_alphas[@]}"; do
                    for policy_seed in "${seeds[@]}"; do
                        for fqe_n_layer in "${fqe_n_hidden_layers[@]}"; do
                            for fqe_embed_size in "${fqe_embed_sizes[@]}"; do
                                echo "new ope-cql training instance start."
                                python ope_tactics_main.py \
                                    --seed 2024 \
                                    --batch_size 256 \
                                    --qf_lr 1e-6 \
                                    --q_n_hidden_layers $fqe_n_layer \
                                    --q_hidden_dims 256 \
                                    --embedding_dim $fqe_embed_size \
                                    --max_timesteps 400000 \
                                    --policy_type "cql" \
                                    --policy_batch_size 256 \
                                    --cql_target_action_gap $cql_target_action_gap \
                                    --cql_tune_init_log_alpha $cql_tune_init_log_alpha \
                                    --policy_embedding_dim $embed_size \
                                    --policy_hidden_dims $hidden_dim \
                                    --policy_n_hidden_layers $n_layer \
                                    --policy_seed $policy_seed \
                                    --policy_path "trained_models" \
                                    --checkpoints_path "evaluation_models_last" \
                                    > ope_logs/train_fqe_cql_fn${fqe_n_layer}_fe${fqe_embed_size}_e${embed_size}_nl${n_layer}_cg${cql_target_action_gap}_ca${cql_tune_init_log_alpha}_s${seed}.log 2>&1 &
                                wait
                            done
                        done
                    done
                done
            done
        done
    done
done


echo "ope-cql training done!"