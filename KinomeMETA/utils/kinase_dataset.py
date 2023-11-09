import os
import torch
from rdkit import Chem
from tqdm import tqdm
import seaborn as sns; sns.set(color_codes=True)

import sys
sys.setrecursionlimit(50000)
import pickle

torch.nn.Module.dump_patches = True

import pandas as pd
#then import my own modules
from AttentiveFP import save_smiles_dicts, get_smiles_dicts, get_smiles_array
from typing import List
import csv
from utils.pretreat_molecule import wash_smiles

def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_task_names(df) -> List[str]:
    """
    Gets the task names from a data CSV file.

    :param df: a pandas DataFrame.
    :return: A list of task names.
    """
    task_names = []
    for col_name in df.columns:
        if (0 in df[col_name].tolist()) or (1 in df[col_name].tolist()):
            task_names.append(col_name)

    return task_names



def get_smiles(path):
    raw_filename = path
    filename = raw_filename.replace('.csv', '')
    feature_filename = raw_filename.replace('.csv', '.pickle')

    smiles_tasks_df = pd.read_csv(raw_filename)
    tasks = get_task_names(smiles_tasks_df)
    print(f'task names are:{tasks}')
    smilesList = smiles_tasks_df.canonical_smiles.values
    print("number of all smiles: ", len(smilesList))
    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in tqdm(smilesList):
        try:
            washed_smi = wash_smiles(smiles)
            mol = Chem.MolFromSmiles(smiles)
            atom_num_dist.append(len(mol.GetAtoms()))
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(washed_smi), isomericSmiles=True))
            remained_smiles.append(smiles)
        except:
            print("not successfully processed smiles: ", smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))
    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["canonical_smiles"].isin(remained_smiles)]

    smiles_tasks_df['cano_smiles'] = canonical_smiles_list
    assert canonical_smiles_list[-1] == Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['cano_smiles'].iat[-1]),
                                                        isomericSmiles=True)
    smilesList = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms()) < 151]

    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb"))
        print('load feature dict successfully')
    else:
        feature_dicts = save_smiles_dicts(smilesList, filename)

    remaining_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    # label_array = remaining_df.iloc[:, 1:-1].values   
    # remain_tasks = []
    # for column, task in zip(label_array.T, tasks):
    #     if (0 in column) & (1 in column):
    #         remain_tasks.append(task)
    # remaining_df_2 = remaining_df[['cano_smiles'] + tasks]
    remaining_df_1 = remaining_df[tasks]

    return remaining_df_1, remaining_df, feature_dicts, canonical_smiles_list

def split_task(task_df, task, random_seed):
    negative_df = task_df[task_df[task] == 0][["cano_smiles", task]]
    positive_df = task_df[task_df[task] == 1][["cano_smiles", task]]
    #negative query and support set
    negative_query = negative_df.sample(frac=1 / 5, random_state=random_seed)
    if negative_query.shape[0] == 0:
        negative_query = negative_df.sample(n=1, replace=True)
    else:
        pass
    negative_support = negative_df.drop(negative_query.index)
    #positive query and support set
    positive_query = positive_df.sample(frac=1 / 5, random_state=random_seed)
    if positive_query.shape[0] == 0:
        positive_query = positive_df.sample(n=1, replace=True)
    else:
        pass
    positive_support = positive_df.drop(positive_query.index)

    support_df = pd.concat([negative_support, positive_support])
    query_df = pd.concat([negative_query, positive_query])

    support_df = support_df.reset_index(drop=True)
    query_df = query_df.reset_index(drop=True)
    return support_df, query_df, positive_query, negative_query






