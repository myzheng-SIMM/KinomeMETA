import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm
import pickle
import torch
import os
import random
import numpy as np

def getProp(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    # print('mw:', mw)
    # print('logp:', logp)

    return mw, logp


def cano_smiles(smile):
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smile), isomericSmiles=True)
    return smi



"""
get short-name

seq_df = pd.read_csv('../data/sequence/kinase_seq_all_label_4.csv', dtype=str)
all_task_df = seq_df[(seq_df['organism'] == 'Homo sapiens') & (seq_df['label2'] == 'isin712') & (seq_df['label4'] != 'other')][['tid_variantid']]
all_task_list = all_task_df['tid_variantid'].tolist()     #527个人类物种

short_name_dict = {}

for i, task in enumerate(all_task_list):
    short_name_df = seq_df[(seq_df['organism'] == 'Homo sapiens') & (seq_df['label2'] == 'isin712') & (seq_df['tid_variantid'] == task)][['short_name']]
    short_name_list = short_name_df['short_name'].tolist()
    short_name = short_name_list[0]
    short_name_dict[task] = short_name

print(short_name_dict)
pickle.dump(short_name_dict, open('../data/new_label_data/short_name_dict.pickle', 'wb'))

"""





"""

get positive smiles - targets

# dataset_all_df = pd.read_csv('../data/new_label_data/tasks_remain_712_pub_chembl_wash.csv')
# seq_df = pd.read_csv('../data/sequence/kinase_seq_all_label_3.csv', dtype=str)
#
# all_task_df = seq_df[(seq_df['organism'] == 'Homo sapiens') & (seq_df['label2'] == 'isin712') & (seq_df['label4'] != 'other')][['tid_variantid']]
# homo_list = all_task_df['tid_variantid'].tolist()     #527 human kinase
#
# for i, task in enumerate(homo_list):
#     positive_df = dataset_all_df[dataset_all_df[task] == 1][['cano_smiles', task]]
#     if i == 0:
#         task_df = positive_df
#     else:
#         task_df = pd.merge(task_df, positive_df, on='cano_smiles', how='outer')
#
# task_df = task_df.rename(columns=short_name_dict)    #convert tid to short name
# pos_smi = task_df.cano_smiles.values     #113268
#
# header_col = 'cano_smiles'
# cols = [x for x in task_df.columns if x != header_col]
# task_pos_smi_df_T = pd.DataFrame(task_df[cols].values.T, columns=task_df[header_col], index=cols)   #smi is columns，target is rows
# print(task_pos_smi_df_T)
# task_pos_smi_df_T.to_csv('../data/new_label_data/task_homo_pos_smi_T.csv')
#
# task_pos_smi_df_T = pd.read_csv('../data/new_label_data/task_homo_pos_smi_T.csv')
# task_pos_smi_df_T.columns = ['task'] + task_pos_smi_df_T.columns[1:].tolist()
# pos_smi = list(task_pos_smi_df_T.columns[1:])
#
# pos_smi_target_dict = {}
# for smi in tqdm(pos_smi):
#     target_df = task_pos_smi_df_T[task_pos_smi_df_T[smi] == 1][['task', smi]]
#     target_list = list(target_df.task.values)
#     pos_smi_target_dict[smi] = target_list
#
# pickle.dump(pos_smi_target_dict, open('../data/new_label_data/pos_smi_target_tasks_dict.pickle', 'wb'))
"""

# short_name_dict = pickle.load(open('../data/new_label_data/short_name_dict.pickle', 'rb'))
# pos_smi_target_dict = pickle.load(open('../data/new_label_data/pos_smi_target_tasks_dict.pickle', 'rb'))



def assign_task_smi(pos_smi_dict_pickle_path, pred_df):
    pos_smi_dict = pickle.load(open(pos_smi_dict_pickle_path, 'rb'))
    both_smi_df = pred_df[pred_df['label'] == 'train'][['cano_smiles']]
    both_smi_list = both_smi_df.cano_smiles.values
    for i, smi in enumerate(both_smi_list):
        try:
            pred_df.loc[pred_df['cano_smiles'] == smi, 'label1'] = str(pos_smi_dict[smi])
        except:pass
    return pred_df


def set_random_seed(seed=0):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use. Default to 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True