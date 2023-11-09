from tqdm import tqdm
import numpy as np
import pandas as pd
import rdkit
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import MolStandardize, AllChem, Descriptors
from collections import defaultdict
import pickle
from utils.kinase_dataset import get_smiles
from utils.result_rename import set_random_seed

mw_bins=[0,
 268.316,
 310.36100000000016,
 340.166,
 365.2640000000001,
 388.4050000000001,
 410.5220000000002,
 433.48700000000014,
 457.5810000000002,
 485.5800000000003,
 522.4400000000002,
 580.6490000000002,
 999.6819999999997]
logp_bins=[-8,
 1.6986499999999998,
 2.617,
 3.238900000000002,
 3.7745000000000015,
 4.303200000000002,
 4.8968000000000025,
 5.711800000000007,
 9.999899999999998]
rot_bins = [-1, 3.0, 4.0, 5.0, 7.0, 9.0, 15.0, 19.0]
hba_bins = [-1, 3.0, 5.0, 7.0, 19.0]
hbd_bins = [-1, 1.0, 2.0, 5.0, 17.0]


def get_class(p,bins):
    for i in range(len(bins) - 1):
        down_t = bins[i]
        up_t = bins[i + 1]
        if p > down_t and p <= up_t:
            return i
    return len(bins)


def getProp(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    return mw, logp, rotb, hbd, hba


def cal_inter_similarity(all_fps, random_sampling=30000):
    if random_sampling:
        ind = np.random.permutation(np.arange(len(all_fps)))[:random_sampling]
        all_fps = [all_fps[i] for i in ind]
    max_sims = []
    min_sims = []
    sims = []
    nfps = len(all_fps)
    for i in tqdm(range(0, nfps-1)):
        tsims = DataStructs.BulkTanimotoSimilarity(all_fps[i], all_fps[i+1:])
        max_sims.append(max(tsims))
        min_sims.append(min(tsims))
        sims.extend(tsims)
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.3)
    sns.distplot(max_sims, ax=axes[0]); axes[0].set_title('max_similarity'); axes[0].set_xticks(np.arange(0,1,0.1))
    sns.distplot(min_sims, ax=axes[1]); axes[1].set_title('min_similarity'); axes[1].set_xticks(np.arange(0,0.12,0.02))
    sns.distplot(sims, ax=axes[2]); axes[2].set_title('similarity'); axes[2].set_xticks(np.arange(0, 1, 0.1))
    plt.savefig('../results/picture/plt_similarity/inter_sim_kinase_RandSele30Kmols')


def cal_pairwise_similarity(support_smiles, query_smiles):
    support_fps, query_fps = [], []
    for smiles in support_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
        except:
            continue
        support_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 512))
    for smiles in query_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
        except:
            continue
        query_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 512))

    sims = []
    for fp1 in query_fps:
        tsims = DataStructs.BulkTanimotoSimilarity(fp1, support_fps)
        sims.extend(tsims)

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_style()
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    sns.distplot(sims, ax=axes)
    axes.set_title('similarity')
    axes.set_xticks(np.arange(0, 1, 0.1))
    plt.savefig('../results/picture/plt_similarity/pos_neg_simi_kinase_RandSele30Kmols')




def mol2decoys(mol, support_fps, decoys_dict, sim_cutoff=0.6, radio = 3):
    mw, logp, rotb, hbd, hba = getProp(mol)
    mw_class = get_class(mw, mw_bins)
    logp_class = get_class(logp, logp_bins)
    decoys = decoys_dict[(mw_class, logp_class)]
    decoys = np.random.permutation(decoys)
    cut_decoys = []
    for decoy in decoys:
        decoys_sims = DataStructs.BulkTanimotoSimilarity(
            AllChem.GetMorganFingerprintAsBitVect(decoy, 2, 512),
            support_fps)
        if max(decoys_sims) < sim_cutoff:
            cut_decoy = Chem.MolToSmiles(decoy)
            cut_decoys.append(cut_decoy)
        if len(cut_decoys) >= radio:
            return cut_decoys
    return cut_decoys


def generate_neg_samples(data_path,save_path,args):
    decoys_from_mv_logp_class = pickle.load(open("./data/negative_sampling/dict_p_bins.pickle", "rb"))
    if isinstance(data_path,pd.DataFrame):
        data_df = data_path
    else:
        data_df_task, data_df_all,_,canonical_smiles_list = get_smiles(data_path)
        data_df = pd.concat([data_df_all['cano_smiles'],data_df_task],axis=1)

    all_smiles = data_df['cano_smiles'].values
    all_pos_smiles = data_df['cano_smiles'][data_df.sum(axis=1) >= 1]  # only use smiles with at least one positive label
    # all_neg_smiles = data_df['canonical_smiles'][data_df.sum(axis=1) == 0]  
    all_fps = []
    for smiles in all_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
        except:
            continue
        all_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 512))


    smiles_decoys_dict = {}
    for smiles in tqdm(all_pos_smiles):
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
        except:
            continue
        cut_decoys = mol2decoys(mol, all_fps, decoys_from_mv_logp_class, sim_cutoff=0.6)
        smiles_decoys_dict[smiles] = cut_decoys

    # pickle.dump(smiles_decoys_dict, open('../data/new_label_data/decoys/smiles_decoys_dict.pickle', "wb"))
    decoys_index = sorted(set(sum(list(smiles_decoys_dict.values()), [])))
    set_random_seed(args.random_seed)
    decoys_index = np.random.permutation(decoys_index)

    decoys_df = pd.DataFrame(columns=data_df.columns[1:], index=decoys_index)
    for task in data_df.columns[1:]:
        task_data_df = data_df[data_df[task] == 1]
        task_pos_smiles = task_data_df['cano_smiles'].dropna().values
        for smiles in tqdm(task_pos_smiles):
            decoy_smiles = smiles_decoys_dict[smiles]
            decoys_df.loc[decoy_smiles, task] = 0

    decoys_df = decoys_df.dropna(axis=0, how='all')
    decoys_df['cano_smiles'] = decoys_df.index
    update_df = pd.concat([data_df, decoys_df], ignore_index=True).rename(columns={'cano_smiles':'canonical_smiles'})
    update_df.to_csv(save_path, index=False)

#generate_neg_samples()





