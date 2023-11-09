import torch
# from rdkit.Chem import rdMolDescriptors, MolSurf
# from rdkit.Chem.Draw import SimilarityMaps
import seaborn as sns; sns.set(color_codes=True)
import numpy as np
import sys
sys.setrecursionlimit(50000)
torch.nn.Module.dump_patches = True
import pandas as pd

from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight
from typing import List
import csv
from utils.kinase_dataset import get_smiles, split_task


class kinaseNshot():
    def __init__(self, path, batchsz, n_way, k_shot, k_query, random_seed):

        self.tasks, self.x, self.feature_dicts, self.canonical_smiles_list = get_smiles(path)

        self.batchsz = batchsz
        self.n_cls = len(self.tasks)
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        self.random_seed = random_seed

        self.x_train_1, self.x_train_2, self.x_train_3, self.x_train_4, self.x_train_5, self.x_train_6, self.x_train_7, self.x_train_8, self.x_train_9, self.x_train_10, self.x_train_11, self.x_train_12, self.x_train_13, self.x_train_14, self.x_train_15, self.x_train_16, self.x_train_17, self.x_train_18, self.x_train_19 = self.tasks.iloc[:, 0:17], self.tasks.iloc[:, 0:34], self.tasks.iloc[:, 0:47], self.tasks.iloc[:, 0:57], self.tasks.iloc[:, 0:67], self.tasks.iloc[:, 0:77], self.tasks.iloc[:, 0:87], self.tasks.iloc[:, 0:97], self.tasks.iloc[:, 0:105], self.tasks.iloc[:, 0:112], self.tasks.iloc[:, 0:136], self.tasks.iloc[:, 0:160], self.tasks.iloc[:, 0:184], self.tasks.iloc[:, 0:203], self.tasks.iloc[:, 0:219], self.tasks.iloc[:, 0:235], self.tasks.iloc[:, 0:251], self.tasks.iloc[:, 0:265], self.tasks.iloc[:, 0:]
        self.x_train_1 = pd.concat([self.x_train_1, self.x['cano_smiles']], axis=1)
        self.x_train_2 = pd.concat([self.x_train_2, self.x['cano_smiles']], axis=1)
        self.x_train_3 = pd.concat([self.x_train_3, self.x['cano_smiles']], axis=1)
        self.x_train_4 = pd.concat([self.x_train_4, self.x['cano_smiles']], axis=1)
        self.x_train_5 = pd.concat([self.x_train_5, self.x['cano_smiles']], axis=1)
        self.x_train_6 = pd.concat([self.x_train_6, self.x['cano_smiles']], axis=1)
        self.x_train_7 = pd.concat([self.x_train_7, self.x['cano_smiles']], axis=1)
        self.x_train_8 = pd.concat([self.x_train_8, self.x['cano_smiles']], axis=1)
        self.x_train_9 = pd.concat([self.x_train_9, self.x['cano_smiles']], axis=1)
        self.x_train_10 = pd.concat([self.x_train_10, self.x['cano_smiles']], axis=1)
        self.x_train_11 = pd.concat([self.x_train_11, self.x['cano_smiles']], axis=1)
        self.x_train_12 = pd.concat([self.x_train_12, self.x['cano_smiles']], axis=1)
        self.x_train_13 = pd.concat([self.x_train_13, self.x['cano_smiles']], axis=1)
        self.x_train_14 = pd.concat([self.x_train_14, self.x['cano_smiles']], axis=1)
        self.x_train_15 = pd.concat([self.x_train_15, self.x['cano_smiles']], axis=1)
        self.x_train_16 = pd.concat([self.x_train_16, self.x['cano_smiles']], axis=1)
        self.x_train_17 = pd.concat([self.x_train_17, self.x['cano_smiles']], axis=1)
        self.x_train_18 = pd.concat([self.x_train_18, self.x['cano_smiles']], axis=1)
        self.x_train_19 = pd.concat([self.x_train_19, self.x['cano_smiles']], axis=1)

        self.indexes = {"train_1": 0, "train_2": 0, "train_3": 0, "train_4": 0, "train_5": 0, "train_6": 0, "train_7": 0, "train_8": 0, "train_9": 0, "train_10": 0, "train_11": 0, "train_12": 0, "train_13": 0, "train_14": 0, "train_15": 0, "train_16": 0, "train_17": 0, "train_18": 0, "train_19": 0}
        self.datasets = {"train_1": self.x_train_1, "train_2": self.x_train_2, "train_3": self.x_train_3, "train_4": self.x_train_4, "train_5": self.x_train_5, "train_6": self.x_train_6, "train_7": self.x_train_7, "train_8": self.x_train_8, "train_9": self.x_train_9, "train_10": self.x_train_10, "train_11": self.x_train_11, "train_12": self.x_train_12, "train_13": self.x_train_13, "train_14": self.x_train_14, "train_15": self.x_train_15, "train_16": self.x_train_16, "train_17": self.x_train_17, "train_18": self.x_train_18, "train_19": self.x_train_19}
        
        self.datasets_cache = {"train_1": self.load_data_cache(self.datasets["train_1"], self.random_seed),  
                               "train_2": self.load_data_cache(self.datasets["train_2"], self.random_seed),
                               "train_3": self.load_data_cache(self.datasets["train_3"], self.random_seed),
                               "train_4": self.load_data_cache(self.datasets["train_4"], self.random_seed),
                               "train_5": self.load_data_cache(self.datasets["train_5"], self.random_seed),
                               "train_6": self.load_data_cache(self.datasets["train_6"], self.random_seed),
                               "train_7": self.load_data_cache(self.datasets["train_7"], self.random_seed),
                               "train_8": self.load_data_cache(self.datasets["train_8"], self.random_seed),
                               "train_9": self.load_data_cache(self.datasets["train_9"], self.random_seed),
                               "train_10": self.load_data_cache(self.datasets["train_10"], self.random_seed),
                               "train_11": self.load_data_cache(self.datasets["train_11"], self.random_seed),
                               "train_12": self.load_data_cache(self.datasets["train_12"], self.random_seed),
                               "train_13": self.load_data_cache(self.datasets["train_13"], self.random_seed),
                               "train_14": self.load_data_cache(self.datasets["train_14"], self.random_seed),
                               "train_15": self.load_data_cache(self.datasets["train_15"], self.random_seed),
                               "train_16": self.load_data_cache(self.datasets["train_16"], self.random_seed),
                               "train_17": self.load_data_cache(self.datasets["train_17"], self.random_seed),
                               "train_18": self.load_data_cache(self.datasets["train_18"], self.random_seed),
                               "train_19": self.load_data_cache(self.datasets["train_19"], self.random_seed)}



    def load_data_cache(self, data_pack, random_seed):
        """
        Collects several batches data for N-shot learning
        """
        #  take 2 way 3 shot as example: 2 * 3
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        for sample in range(50):  # num of episodes
            batch_list = []
            task_list = data_pack.columns[:-1]
            for counter in range(0, len(task_list), self.batchsz):
                batch = np.array(data_pack.columns[:-1][counter:min(counter + self.batchsz, len(task_list))])
                batch_list.append(batch)

            for i, eval_batch in enumerate(batch_list):
                batch_df = pd.concat([data_pack.loc[:, eval_batch], data_pack['cano_smiles']], axis=1)
                batch_task = batch_df.columns[:-1]

                support_x = np.empty((len(batch_task), setsz, 1), dtype='object')
                support_y = np.empty((len(batch_task), setsz), dtype=np.int)
                query_x = np.empty((len(batch_task), querysz, 1), dtype='object')
                query_y = np.empty((len(batch_task), querysz), dtype=np.int)


                for j, task in enumerate(batch_task):
                    support_df, _, _, _ = split_task(batch_df, task, random_seed)

                    support_neg_df = support_df[support_df[task] == 0][["cano_smiles", task]]
                    support_pos_df = support_df[support_df[task] == 1][["cano_smiles", task]]
                    #negative query set and support set of support set
                    query_sup_neg = support_neg_df.sample(frac=1 / 4, random_state=random_seed)
                    if query_sup_neg.shape[0] == 0:
                        query_sup_neg = support_neg_df.sample(n=1, replace=True)
                    else:
                        pass
                    support_sup_neg = support_neg_df.drop(query_sup_neg.index)
                    #positive query set and support set of support set
                    query_sup_pos = support_pos_df.sample(frac=1 / 4, random_state=random_seed)
                    if query_sup_pos.shape[0] == 0:
                        query_sup_pos = support_pos_df.sample(n=1, replace=True)
                    else:
                        pass
                    support_sup_pos = support_pos_df.drop(query_sup_pos.index)
                    
                    # if support_sup_pos.shape[0] == 0:
                    #     support_sup_pos = query_sup_pos

                    support_sup_neg = support_sup_neg.reset_index(drop=True)
                    support_sup_pos = support_sup_pos.reset_index(drop=True)
                    query_sup_neg = query_sup_neg.reset_index(drop=True)
                    query_sup_pos = query_sup_pos.reset_index(drop=True)

                    support_sup = pd.concat([support_sup_pos, support_sup_neg])
                    support_que = pd.concat([query_sup_pos, query_sup_neg])

                    # sampling in support set of support set
                    if support_sup_pos.shape[0] < self.k_shot:
                        selected_sup_pos = np.random.choice(support_sup_pos.shape[0], self.k_shot, True)
                    else:
                        selected_sup_pos = np.random.choice(support_sup_pos.shape[0], self.k_shot, False)
                    selected_sup_neg = np.random.choice(range(support_sup_pos.shape[0], support_sup.shape[0]), self.k_shot, False)
                    selected_train = np.vstack([selected_sup_pos, selected_sup_neg]).flatten()   
                    # sampling in query set set of support set
                    selected_que_neg = np.random.choice(query_sup_neg.shape[0], self.k_query, False)
                    selected_que_pos = np.random.choice(range(query_sup_pos.shape[0], support_que.shape[0]), self.k_query, False)
                    selected_test = np.vstack([selected_que_pos, selected_que_neg]).flatten()


                    # meta-training, select the first k_shot samples for each class as support samples
                    for offset, smiles_idx in enumerate(selected_train):

                        support_x[j, offset] = support_sup.iloc[[smiles_idx]].cano_smiles.values
                        support_y[j, offset] = support_sup[task].iloc[[smiles_idx]].values  # relative indexing


                    # meta-test, treat following k_query samples as query samples
                    for offset, smiles_idx in enumerate(selected_test):
                        query_x[j, offset] = support_que.iloc[[smiles_idx]].cano_smiles.values
                        query_y[j, offset] = support_que[task].iloc[[smiles_idx]].values  # relative indexing
                s_x_atom, s_x_bonds, s_x_atom_index, s_x_bond_index, s_x_mask, s_smiles_to_rdkit_list = get_smiles_array(
                        support_x.ravel(), self.feature_dicts)
                support_x_atom = torch.tensor(s_x_atom.reshape((len(batch_task), setsz, -1, s_x_atom.shape[-1])))
                support_x_bond = torch.tensor(s_x_bonds.reshape((len(batch_task), setsz, -1, s_x_bonds.shape[-1])))
                support_x_atom_index = torch.LongTensor(
                        s_x_atom_index.reshape((len(batch_task), setsz, -1, s_x_atom_index.shape[-1])))
                support_x_bond_index = torch.LongTensor(
                        s_x_bond_index.reshape((len(batch_task), setsz, -1, s_x_bond_index.shape[-1])))
                support_x_mask = torch.tensor(s_x_mask.reshape((len(batch_task), setsz, s_x_mask.shape[-1])))
                
                support_x = (support_x_atom, support_x_bond, support_x_atom_index, support_x_bond_index, support_x_mask)

                q_x_atom, q_x_bonds, q_x_atom_index, q_x_bond_index, q_x_mask, q_smiles_to_rdkit_list = get_smiles_array(
                        query_x.ravel(), self.feature_dicts)

                query_x_atom = torch.tensor(q_x_atom.reshape((len(batch_task), querysz, -1, q_x_atom.shape[-1])))
                query_x_bond = torch.tensor(q_x_bonds.reshape((len(batch_task), querysz, -1, q_x_bonds.shape[-1])))
                query_x_atom_index = torch.LongTensor(
                        q_x_atom_index.reshape((len(batch_task), querysz, -1, q_x_atom_index.shape[-1])))
                query_x_bond_index = torch.LongTensor(
                        q_x_bond_index.reshape((len(batch_task), querysz, -1, q_x_bond_index.shape[-1])))
                query_x_mask = torch.tensor(q_x_mask.reshape((len(batch_task), querysz, q_x_mask.shape[-1])))
                    # query_x_smiles_to_rdkit_list = q_smiles_to_rdkit_list.reshape(self.batchsz, querysz, -1,)
                query_x = (query_x_atom, query_x_bond, query_x_atom_index, query_x_bond_index, query_x_mask)
                data_cache.append([support_x, support_y, query_x, query_y])
        print('complete')
        return data_cache

    def __get_batch(self, mode):
        """
        Gets next batch from the dataset with name.
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode], self.random_seed)

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

    def get_batch(self, mode):

        """
        Get next batch
        :return: Next batch
        """
        x_support_set, y_support_set, x_target, y_target = self.__get_batch(mode)


        return x_support_set, y_support_set, x_target, y_target









