from argparse import Namespace,ArgumentParser

def meta_train_parse_args()  -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--meta_batchsz",type=int,default=15,help='the meta-learning batch size in meta-training')
    parser.add_argument("--meta_lr",type=float,default=1e-3,help='the meta-learning learning rate in meta-training')
    parser.add_argument("--num_updates",type=int,default=5,help='the update times of the meta-model in meta-training')
    parser.add_argument("--random_seed",type = int ,default=68,help='the random seed')

    parser.add_argument("--p_dropout",type=float,default=0.2,help='dropout rate of model in finetune process,usually the same as training')
    parser.add_argument("--fingerprint_dim",type = int ,default=250,help= 'fingerprint dimension of Attentive model in finetune process,usually the same as training')
    parser.add_argument("--radius",type = int ,default=3,help='the radius of Attentive model in finetune process,usually the same as training')
    parser.add_argument("--T",type = int ,default=2,help='the readout steps of Attentive model in finetune process,usually the same as training')
    
    parser.add_argument("--n_way",type = int ,default=2,help='the classes of samples in task')
    parser.add_argument("--k_shot",type = int ,default=3,help='the sample numbers in one class of a task')
    parser.add_argument("--k_query",type = int ,default=1,help='the sample numbers in query set')

    parser.add_argument('--device',type = int, default=0,help='Which GPU to use')

    parser.add_argument('--train_path',type=str,default='data/train/decoys/clus_train_tasks_434_decoys.csv',help='kinases label data with decoys for training')
    parser.add_argument('--save_dir',type=str,default='./results/meta-learner/',help='default directory to save meta-learner')
    
    args = parser.parse_args()
    return args


def finetune_parse_args() -> Namespace:
    parser = ArgumentParser()
    #toy_finetune_task.csv  validtasks_183.csv     ./results/meta-learner/2022-08-24-04_18_42_meta-learner_state_dict.pth    base-learner/104319_state_dict.pth
    parser.add_argument('--kinase_data_path',type=str,default="./data/finetune/toy_finetune_task.csv",help='the label data path of kinase tasks for fine-tuning')
    parser.add_argument('--meta_model_path',type = str,default = './results/meta-learner/2022-08-24-04_18_42_meta-learner_state_dict.pth',help='the best meta-learner state_dict')
    parser.add_argument('--save_dir',type=str,default=None,help='default save directory of fine-tuned base-learner models. When continuing to fine-tune a base-learner model, you can pass the corresponding base-learner path as well,but can only fine-tune one task at one time')

    parser.add_argument('--no_decoys',type=bool,default=False)   #,action='store_true')

    parser.add_argument("--p_dropout",type=float,default=0.2,help='dropout rate of model in finetune process,usually the same as training')
    parser.add_argument("--fingerprint_dim",type = int ,default=250,help= 'fingerprint dimension of Attentive model in finetune process,usually the same as training')
    parser.add_argument("--radius",type = int ,default=3,help='the radius of Attentive model in finetune process,usually the same as training')
    parser.add_argument("--T",type = int ,default=2,help='the readout steps of Attentive model in finetune process,usually the same as training')
    parser.add_argument("--random_seed",type = int ,default=68,help='the random seed')
    parser.add_argument("--num_train",type=int,default=50,help='the epoch numbers for finetune process')
    parser.add_argument("--batch_size",type=int,default=5,help='batch size in finetune process')
    parser.add_argument('--device',type = int, default=0,help='Which GPU to use')

    args = parser.parse_args()
    return args



def predict_parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--random_seed",type = int ,default=68,help='the random seed')
    parser.add_argument('--compound_path',type=str,default='./data/compounds/toy_compounds.csv',help='The path of the compound file to be predicted, including the smiles of the predicted molecules with the column name smiles')
    parser.add_argument('--baselearner_dir',type=str,default='./results/base-learner/',help='path contained fine-tuned base-learner models for prediction. Users can adjust the models included in the path to obtain the desired prediction range.')
    parser.add_argument('--save_path',type=str,default=None,help='default save path with task time is the directory, "./results/prediction/"')

    parser.add_argument('--device',type = int, default=0,help='Which GPU to use')
    parser.add_argument('--batch_size',type = int,default=5,help='batch size in prediction')


    args = parser.parse_args()
    return args