import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from utils.pretreat_molecule import wash_smiles

def BC(x):
    try:
        count = x.shape[0]    
        g = x.skew()          
        k = x.kurt()          
        bc = (g ** 2 + 1 ) / (k + (3 * (count - 1) ** 2) / ((count - 2) * (count - 3)))
    except:
        print(f'bimodal coefficient cannot be calculated')
        bc = None
    return bc


def count_bc_hit_rate(data):
    count = 0
    for i in data:
        try:
            if i >= 0.5:
                count = count + 1
            else:
                pass
        except:
            print(i)
    hit_rate = count/len(data)
    bc = BC(data)
    return hit_rate, bc


def assign_bc_hit_label(data_pred, dataset_smi,pred_position):
    hit_result_list = []
    bc_result_list = []
    num_compounds = len(data_pred[0:])
    print('compound num:' + str(num_compounds))
    for i in range(num_compounds):
        pred = data_pred.iloc[i, pred_position:]
        hit, bc = count_bc_hit_rate(pred)
        hit_result_list.append(hit)
        bc_result_list.append(bc)
    #prediction hit rate and bimodal coefficient
    data_pred['hit'] = hit_result_list
    data_pred['bc'] = bc_result_list

    target_smi = data_pred.cano_smiles.values
    both_smi = list(set(target_smi) & set(dataset_smi))
    #label smis for training 
    data_pred['label'] = None
    for i, smi in enumerate(both_smi):
        data_pred.loc[data_pred['cano_smiles'] == smi, 'label'] = 'train'
    return data_pred

