#!/bin/bash

python finetune.py \
--kinase_data_path "./data/finetune/toy_finetune_task.csv" \
--meta_model_path './results/meta-learner/2022-08-24-04_18_42_meta-learner_state_dict.pth' \
--save_dir "./results/base-learner/toy_finetune_task/" \
--random_seed 68 \
--num_train 50 \
--device 0 


#the other args of models usually are set same as those in training. 

