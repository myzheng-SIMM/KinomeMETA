#!/bin/bash

python predict.py \
--compound_path './data/compounds/toy_compounds.csv' \
--baselearner_dir "./results/base-learner/" \
--device 0 \
--batch_size 5