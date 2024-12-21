# Offline RL for Badminton Tactical Decision-Making
This repo is the official implementation for the paper "Offline Reinforcement Learning for Badminton Tactical Decision-Making".

## Environment setup
1. Install and configure anaconda
2. Create a new conda env: `conda env create -f conda_env.yaml`

## Dataset
We integrated two badminton datasets ShuttleSet and ShuttleSet22 into a larger dataset referred to as Shuttle. The raw data can ba found in the folder *"data/shuttle"*.

Finally, we generated two kinds of datasets: *shuttle_both_agent_#.pkl* and *shuttle_sequence_#.pkl* for player-based MDP and turn-based sequence decision-making, respectively.

## Training
We provide the code for training preference-based reward model and Offline RL policies:

- *"reward_learn_main.py"*: main fucntion for training preference-based reard model
- *"cql_tactics_main.py"*: main function for training CQL with Hybrid Action Sapce and BC policies
- *"dt_tactics_main.py"*: main function for training DT and Sequence-based BC policies

You can directly run the sollowing shell scripts to start training:

```bash
bash reward_learn_train_final.sh
bash cql_train_final.sh
bash mlp_based_bc_train_final.sh
bash dt_train_final.sh
bash dt_based_bc_train_final.sh
```

## Evaluation
We provide the code for policy evaluation:

- *"tactics_model_evaluation.py"*: main function for reward-model-based policy evaluation
- *"fqe_tactics_main.py"*: main function for FQE-based policy evaluation
- *"domain_metrics_tactics_main.py"*: main function for domain-metric-based policy evaluation

You can directly run the sollowing shell scripts to evaluate:
```bash
bash cql/mlp_based_bc/dt/dt_based_bc_evaluation_final.sh
bash fqe_cql/mlp_based_bc/dt/dt_based_bc_train_final.sh
bash cql/mlp_based_bc/dt/dt_based_bc_domain_metrics_final.sh
```

## Citation


## LICENSE
MIT License

## Some fancy works to inspire future research
- Learning Physically Simulated Tennis Skills from Broadcast Videos. TOG. 2023.
- SMPLOlympics: Sports Environments for Physically Simulated Humanoids. Arxiv. 2024.
