import sys
from Reptile.meta import MetaLearner
from AttentiveFP.AttentiveLayers import Fingerprint
import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from utils.data_split import kinaseNshot
import pickle
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from utils.kinase_dataset import get_smiles
from AttentiveFP import get_smiles_array
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from utils.parse_args import meta_train_parse_args
from utils.result_rename import set_random_seed


def save_model(model: nn.Module, save_path: str, k_shot, meta_batchsz):
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    name = f"{now}_meta-learner"
    torch.save(
        model.state_dict(),
        os.path.join(save_path, f"{name}_state_dict.pth"),
    )

def main(args):
    meta_batchsz = args.meta_batchsz
    meta_lr = args.meta_lr
    num_updates = args.num_updates
    random_seed = args.random_seed
    p_dropout = args.p_dropout
    fingerprint_dim = args.fingerprint_dim
    radius = args.radius
    T = args.T
    per_task_output_units_num = 2  # for classification model with 2 classes
    output_units_num = per_task_output_units_num

    n_way = args.n_way
    k_shot = args.k_shot 
    k_query = args.k_query  
    device = torch.device(args.device)

    path = args.train_path
    save_path = args.save_dir


    #data preparation
    kinase_data_remained = kinaseNshot(path, batchsz=meta_batchsz, k_shot=k_shot, k_query=k_query, n_way=n_way, random_seed=random_seed)
    feature_dicts, canonical_smiles_list = kinase_data_remained.feature_dicts, kinase_data_remained.canonical_smiles_list

    set_random_seed(args.random_seed)
    #Model instantiation
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([canonical_smiles_list[0]], feature_dicts)
    num_atom_features = x_atom.shape[-1]
    num_bond_features = x_bonds.shape[-1]


    meta = MetaLearner(Fingerprint, (radius, T, num_atom_features,num_bond_features,
            fingerprint_dim, output_units_num, p_dropout), n_way=n_way, k_shot=k_shot, k_query=k_query, meta_batchsz=meta_batchsz, meta_lr=meta_lr,num_updates=num_updates).to(device)



    # main loop
    list_train_roc = []
    list_train_loss = []
    list_test_roc = []
    list_test_loss = []
    episode = []
    for episode_num in range(2850):
        train_losses = []
        test_losses = []
        if episode_num < 150:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_1')
        elif 149 < episode_num < 300:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_2')
        elif 299 < episode_num < 450:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_3')
        elif 449 < episode_num < 600:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_4')
        elif 599 < episode_num < 750:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_5')
        elif 749 < episode_num < 900:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_6')
        elif 899 < episode_num < 1050:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_7')
        elif 1049 < episode_num < 1200:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_8')
        elif 1199 < episode_num < 1350:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_9')
        elif 1249 < episode_num < 1500:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_10')
        elif 1499 < episode_num < 1650:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_11')
        elif 1649 < episode_num < 1800:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_12')
        elif 1799 < episode_num < 1950:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_13')
        elif 1949 < episode_num < 2100:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_14')
        elif 2099 < episode_num < 2250:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_15')
        elif 2249 < episode_num < 2400:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_16')
        elif 2399 < episode_num < 2550:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_17')
        elif 2549 < episode_num < 2700:
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_18')
        elif 2699 < episode_num :
            support_x, support_y, query_x, query_y = kinase_data_remained.get_batch('train_19')
        support_x = [x.to(device) for x in support_x]
        query_x = [x.to(device) for x in query_x]
        support_y = torch.tensor(support_y).to(device)
        query_y = torch.tensor(query_y).to(device)

        # backprop has been embeded in forward func.
        train_loss, rocs = meta(support_x, support_y, query_x, query_y)
        train_avg_roc = np.array(rocs).mean()
        for i in range(len(train_loss)):
            train_loss1 = train_loss[i].cpu().detach().numpy()
            train_losses.append(train_loss1)

        test_loss, test_acc, test_pre_score, test_recall_score, test_mcc_score, test_roc_score, test_f1_score = meta.pred(support_x, support_y, query_x, query_y)

        for i in range(len(test_loss)):
            test_loss1 = test_loss[i].cpu().detach().numpy()
            test_losses.append(test_loss1)

        train_avg_loss = np.array(train_losses).mean()
        test_avg_loss = np.array(test_losses).mean()

        results = [episode_num, train_avg_loss, test_avg_loss, test_acc, test_pre_score, test_recall_score, test_mcc_score, test_roc_score, test_f1_score]

        with open(os.path.join(save_path,'log.txt'), 'a') as f:
            f.write(','.join([str(r) for r in results]))
            f.write('\n')

        list_test_loss.append(test_avg_loss)
        list_test_roc.append(test_roc_score)
        list_train_loss.append(train_avg_loss)
        list_train_roc.append(train_avg_roc)

        print('episode:', episode_num, '\ttrain roc:%.6f' % train_avg_roc,
              '\t\ttrain_loss:%.6f' % train_avg_loss,
              '\t\tvalid acc:%.6f' % test_acc, '\t\tloss:%.6f' % test_avg_loss,
              '\t\tvalid pre:%.6f' %test_pre_score, '\t\tvalid recall:%.6f' %test_recall_score,
              '\t\tvalid mcc:%.6f' %test_mcc_score, '\t\tvalid roc:%.6f' %test_roc_score, '\t\tvalid f1:%.6f' %test_f1_score)
        episode.append(episode_num)

        #code to save model
        # if episode_num == 149:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 299:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 449:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 599:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 749:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 899:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 1049:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 1199:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 1349:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 1499:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 1649:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 1799:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 1949:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 2099:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 2249:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 2399:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 2549:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')
        # elif episode_num == 2699:
        #     save_model(model=meta.learner.net, save_path=save_path)
        #     print('save seq_meta-learner')


    save_model(model=meta.learner.net, save_path=save_path, k_shot=k_shot, meta_batchsz=meta_batchsz)
    print('save final meta-learner')



if __name__ == '__main__':
    args = meta_train_parse_args()
    set_random_seed(args.random_seed)
    main(args)
