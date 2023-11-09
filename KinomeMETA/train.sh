#!/bin/bash

python train.py \
--meta_batchsz 15 \
--meta_lr 0.001 \
--num_updates 5 \
--random_seed 68 \
--p_dropout 0.2 \
--fingerprint_dim 250 \
--radius 3 \
--T 2 \
--n_way 2 \
--k_shot 3 \
--k_query 1 \
--device 0 \
--train_path 'data/train/decoys/clus_train_tasks_434_decoys.csv' \
--save_dir './results/meta-learner/' 


