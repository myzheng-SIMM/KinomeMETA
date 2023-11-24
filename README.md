# KinomeMETA
The code of "KinomeMETA: meta-learning enhanced kinome-wide polypharmacology profiling"

Kinase inhibitors are crucial in cancer treatment, but drug resistance and side effects hinder the development of effective drugs. To address these challenges, it is essential to analyze the polypharmacology of kinase inhibitor and identify compound with high selectivity profile. This study presents KinomeMETA, a framework for profiling the activity of small molecule kinase inhibitors across a panel of 661 kinases. By training a meta-learner based on a graph neural network and fine-tuning it to create kinase-specific learners, KinomeMETA outperforms benchmark multi-task models and other kinase profiling models. It provides higher accuracy for understudied kinases with limited known data and broader coverage of kinase types, including important mutant kinases. Case studies on the discovery of new scaffold inhibitors for PKMYT1 and selective inhibitors for FGFRs demonstrate the role of KinomeMETA in virtual screening and kinome-wide activity profiling. Overall, KinomeMETA has the potential to accelerate kinase drug discovery by more effectively exploring the kinase polypharmacology landscape.

### Train model
```
bash train.sh

or

python train.py   --meta_batchsz 15 \
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
```

### Finetune new kinase models
```
bash finetune.sh

or

python finetune.py  --kinase_data_path "./data/finetune/toy_finetune_task.csv" \
                    --meta_model_path './results/meta-learner/2022-08-24-04_18_42_meta-learner_state_dict.pth' \
                    --save_dir "./results/base-learner/toy_finetune_task/" \
                    --random_seed 68 \
                    --num_train 50 \
                    --device 0 
```

### Predict new compounds
```
bash predict.sh

or

python predict.py --compound_path './data/compounds/toy_compounds.csv' \
                  --baselearner_dir "./results/base-learner/" \
                  --device 0 \
                  --batch_size 5
```

### Setup and dependencies
environment.yaml contains environment of this project.

### Requirements
python = 3.6.13  
pytorch = 1.10.0
scikit-learn = 0.24.2  
numpy = 1.19.5
scipy = 1.5.4
rdkit = 2020.09.1.0
