import pickle
from AttentiveFP.AttentiveLayers import Fingerprint
import torch
import pandas as pd
from torch import optim
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import os
import torch
from rdkit import Chem
import seaborn as sns; sns.set(color_codes=True)
import sys
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem as ch
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
import time
sys.setrecursionlimit(50000)
torch.nn.Module.dump_patches = True
#then import my own modules
from AttentiveFP import save_smiles_dicts, get_smiles_dicts, get_smiles_array
from typing import List
import csv
from tqdm import tqdm
from utils.result_rename import assign_task_smi,set_random_seed
from utils.pretreat_molecule import wash_smiles
from utils.bimodal_coefficient import assign_bc_hit_label
from utils.parse_args import predict_parse_args

#function to wash smiles and obtain compound features
def get_washed_smiles(path):
    """
    This function is used to get valid smiles.
    The provided smiles must be covertible to mol object by RDKit and able to get atom nums.
    The predictable compounds should be less than 151 atoms.
    """
    raw_filename = path
    filename = raw_filename.replace('.csv', '')
    feature_filename = raw_filename.replace('.csv', '.pickle')
    smiles_tasks_df = pd.read_csv(raw_filename)  #.iloc[:10]
    smiles_tasks_df = smiles_tasks_df.dropna(subset=['smiles'])
    smilesList = smiles_tasks_df.smiles.values
    print("number of all smiles: ", len(smilesList))
    atom_num_dist = []
    remained_smiles = []
    smilist = []
    smiles_wash = []
    canonical_smiles_list = []
    for smi in tqdm(smilesList):
        try:
            smiles = wash_smiles(smi)
            mol = Chem.MolFromSmiles(smi)
            atom_num_dist.append(len(mol.GetAtoms()))
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))  #use the washed smiles
            remained_smiles.append(smi)
            smiles_wash.append(smiles)            
        except:
            print("not successfully processed smiles: ", smi)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))

    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]  # the remaining smiles
    smiles_tasks_df['smiles_wash'] = smiles_wash
    smiles_tasks_df['cano_smiles'] = canonical_smiles_list
    # check the canonical smiles and reserve molecules with atomic numbers less than 151
    print('keep smiles with atomic numbers less than 151')
    for smi in tqdm(canonical_smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol == None:
            pass
        elif len((mol).GetAtoms()) < 151:
            smilist.append(smi)

    # get feature_dicts of valid smiles
    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb"))
        print('load feature dict successfully')
    else:
        feature_dicts = save_smiles_dicts(smilist, filename)
    remaining_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    return remaining_df, feature_dicts, smilist

#function to obtain task model information, this function needs to be modified when expanding the prediction range
def get_task_info(task_range,others:List = None):
    all_task_list,short_name_dict = [],{}

    if task_range == 'all661':
        seq_df = pd.read_excel('./data/kinase_sequence/SI-KinomeMETA-Table-S5, S6, S7.xlsx', dtype=str,sheet_name=0)
        for index in seq_df.index:
            tid , variant_id,short_name = seq_df.at[index,'tid_x'],seq_df.at[index,'variant_id'],seq_df.at[index,'short_name']
            if pd.isna(variant_id):
                taskid = tid
            elif variant_id == 'V564F':
                taskid = 'FGFR2.V564F_6'
            else:
                taskid = tid + "_" + variant_id + '.0'  
            if taskid == '11925_1518.0':taskid="FGFR4.V550L_6"
            all_task_list.append(taskid)
            short_name_dict[taskid]=short_name

    else:
        print('The task range is mismatched')
        all_task_list = []
    
    if others is not None:
        all_task_list.extend(others)

    return all_task_list,short_name_dict


#start
tasktime = time.strftime(f"%Y%m%d_%H%M%S",time.strptime(time.ctime(time.time()+8*60*60)))
print('\n---------------current prediction task time:',tasktime)

args = predict_parse_args()
set_random_seed(args.random_seed)
#Path of smiles file to be predicted. The column of the smiles column should be 'smiles'
path =args.compound_path    #'/home/house/quning/jupyter/data_process/GDSC_HiDRA/ProcessedFile/TCGA/kinome/TCGA_drugs_to_kinome.csv'                                    #args.compound_path  #  


#prediction task info, existed models use tid_variantid as index, new models can pass into all_task_list by the variable others with list type
task_range ="all661" #'all661'
all_task_list,short_name_dict = get_task_info(task_range)     #get kinase name mapping dict for 661 kinases in our article
if 'old' in task_range:
    short_name_pickle_path = './data/short_name_dict.pickle'
    short_name_dict = pickle.load(open(short_name_pickle_path,'rb'))

all_task_list =[task[:-15] for task in os.listdir(args.baselearner_dir) if '_state_dict.pth' in task and task[:-15] in all_task_list]


#save_path
if args.save_path is None:
    args.save_path = f'./results/prediction/prediction_{tasktime}_{os.path.splitext(os.path.basename(path))[0]}.xlsx'   #{os.path.basename(path)}'
save_path = args.save_path  #'/home/house/quning/jupyter/data_process/GDSC_HiDRA/ProcessedFile/TCGA/kinome/pred_TCGA.csv'     

#args, should be same as training args
p_dropout = 0.2
fingerprint_dim = 250
radius = 3
T = 2
batch_size = args.batch_size
per_task_output_units_num = 2  # for classification model with 2 classes
output_units_num = per_task_output_units_num
device = torch.device(args.device)


#mapping file
pos_smi_dict_pickle_path = './data/mapping/pos_smi_target_tasks_dict.pickle'    #label positive targets of compounds in the training dataset    
dataset_smiles_all_wash = pickle.load(open('./data/mapping/dataset_smiles_all_wash.pickle', 'rb'))    #annote whether the smi is in the training dataset


#get washed smiles
remaining_df, feature_dicts, canonical_smiles_list = get_washed_smiles(path)
remaining_df = remaining_df.dropna(subset=['cano_smiles']) 
remaining_df = remaining_df.reset_index(drop=True)
pred_position = remaining_df.shape[1]

#Model instantiation
x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([canonical_smiles_list[0]], feature_dicts)
num_atom_features = x_atom.shape[-1]
num_bond_features = x_bonds.shape[-1]
model = Fingerprint(radius, T, num_atom_features, num_bond_features, fingerprint_dim, output_units_num, p_dropout)



#prediction by tasks and smiles
fail_task = []
for i, task in enumerate(tqdm(all_task_list)):
    try:
        model_path = os.path.join(args.baselearner_dir,task+'_state_dict.pth')
        model.load_state_dict(torch.load(model_path,map_location='cpu'))

        model.eval()
        model.to(device)

        smileslist = remaining_df['cano_smiles']
        batch_list = []
        pred_list = []

        for i in range(0, smileslist.shape[0], batch_size):
            batch = smileslist[i:i + batch_size]
            batch_list.append(batch)
        for counter, smilist in enumerate(batch_list):
            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smilist, feature_dicts)
            x_atom = torch.Tensor(x_atom).to(device)
            x_bonds = torch.Tensor(x_bonds).to(device)
            x_atom_index = torch.LongTensor(x_atom_index).to(device)
            x_bond_index = torch.LongTensor(x_bond_index).to(device)
            x_mask = torch.Tensor(x_mask).to(device)
            y_pred = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
            pred_ = list(F.softmax(y_pred, dim=-1).data.cpu().numpy()[:, 1])
            pred_list.extend(pred_)

        pred_dict = {'pred': pred_list}
        pred_df = pd.DataFrame(pred_dict)
        remaining_df[task] = pred_df
    except Exception as e:
        print(f'an error occured, {e}, {task} predict fail')
        fail_task.append(task)


    remaining_df.to_csv(save_path, index=False) #save after one task finished 

print('remaining_df shape: ',pred_position,remaining_df.shape)
print('predict success!')

#annotate information for smiles
def assign_label(pred_df, dataset_smiles_all_wash, short_name_dict, pos_smi_dict_pickle_path,pred_position):
    pred_df_1 = pred_df   #assign_bc_hit_label(pred_df, dataset_smiles_all_wash,pred_position)
    print('assign bc hit and label success!')
    pred_df_2 = pred_df_1.rename(columns=short_name_dict)
    print('rename success!')
    pred_df_3 = pred_df_2 #assign_task_smi(pos_smi_dict_pickle_path, pred_df_2)
    print('assign task names success!')
    return pred_df_3

final_pre_result = assign_label(remaining_df, dataset_smiles_all_wash, short_name_dict, pos_smi_dict_pickle_path,pred_position)
print(f'prediction finished, save at: {save_path}')
final_pre_result.to_excel(save_path, index=False)

