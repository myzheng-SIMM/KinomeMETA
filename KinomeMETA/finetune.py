import pickle
from AttentiveFP.AttentiveLayers import Fingerprint
from AttentiveFP import get_smiles_array
import torch
import pandas as pd
from torch import optim
import os
import torch.nn as nn
from utils.metrics import recall, precision, mcc, roc, accuracy, f1, bacc
import numpy as np
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from utils.kinase_dataset import get_smiles, split_task
import copy
from utils.parse_args import finetune_parse_args
from utils.negative_sampling import generate_neg_samples
from utils.result_rename import set_random_seed
import warnings

#args
args = finetune_parse_args()
p_dropout = args.p_dropout
fingerprint_dim = args.fingerprint_dim
radius = args.radius
T = args.T
per_task_output_units_num = 2  # for classification model with 2 classes
output_units_num = per_task_output_units_num
random_seed = args.random_seed
num_train =args.num_train
batch_size = args.batch_size

device = torch.device(args.device)
set_random_seed(args.random_seed)


#get the corresponding path and generate decoys file
path_result = args.kinase_data_path
path_decoys = os.path.join(os.path.dirname(args.kinase_data_path),'decoys',(os.path.basename(args.kinase_data_path).split(".")[0] + '_decoys' + '.csv'))
if not os.path.exists(path_decoys) and not args.no_decoys:
    os.makedirs(os.path.dirname(path_decoys),exist_ok=True)
    generate_neg_samples(path_result,path_decoys,args)


#model save dir after finetune
if args.save_dir is None:
    default_save_dir = "./results/base-learner/"
    args.save_dir = os.path.join(default_save_dir,os.path.basename(path_result).split(".")[0]) 
    if args.no_decoys:
        args.save_dir = os.path.join(args.save_dir,'nodecoys')
        warnings.warn("You have chosen no decoys, please ensure that there are negative data in the raw data")
save_dir = args.save_dir 
os.makedirs(save_dir,exist_ok=True)
print(f'save directory at {save_dir}')


#processing data, usually no need to modify
task_df_info, task_df_all, feature_dicts_all, real_canonical_smiles_list = get_smiles(path_result)
if not args.no_decoys:
    _, task_df_decoys, feature_dicts, canonical_smiles_list = get_smiles(path_decoys)
else:
    task_df_decoys, feature_dicts, canonical_smiles_list = task_df_all, feature_dicts_all, real_canonical_smiles_list

tasks_name = task_df_all.columns[(task_df_all.shape[1]-task_df_info.shape[1]-1):-1].tolist()  #skip the original file columns and the new append 'cano_smiles' column 


#Model instantiation
x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([canonical_smiles_list[0]],
                                                                                             feature_dicts)
num_atom_features = x_atom.shape[-1]
num_bond_features = x_bonds.shape[-1]
model = Fingerprint(radius, T, num_atom_features, num_bond_features, fingerprint_dim, output_units_num, p_dropout)
model.to(device)


#function to save model
def save_model(model: nn.Module, save_path: str, task_name:str):
    name = f"{task_name}"
    torch.save(
        model.state_dict(),
        os.path.join(save_path, f"{name}_state_dict.pth"),
    )

# function to train one epoch
def train(model, task, dataset, optimizer, loss_function):
    model.train()
    valList = np.arange(0, dataset.shape[0])
    # shuffle them
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch, :]
        smiles_list = batch_df['cano_smiles']
        y_target = torch.cuda.LongTensor(batch_df[task].values)
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list, feature_dicts)
        x_atom = torch.Tensor(x_atom).to(device)
        x_bonds = torch.Tensor(x_bonds).to(device)
        x_atom_index = torch.LongTensor(x_atom_index).to(device)
        x_bond_index = torch.LongTensor(x_bond_index).to(device)
        x_mask = torch.Tensor(x_mask).to(device)
        y_pred_ = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)

        y_target = y_target.cpu().detach().numpy()
        validInds = np.where((y_target == 0) | (y_target == 1))[0]
        y = np.array([y_target[v] for v in validInds]).astype(float)
        validInds = torch.cuda.LongTensor(validInds).squeeze().to(y_pred_.device)
        y_pred = torch.index_select(y_pred_, 0, validInds)

        loss = loss_function(y_pred, torch.cuda.LongTensor(y).to(y_pred_.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# function to evaluation
def eval(model, task, dataset, loss_function):
    model.eval()
    valList = np.arange(0, dataset.shape[0])
    # shuffle them
    np.random.shuffle(valList)
    batch_list = []
    y_val_list = []
    y_pred_list = []
    losses_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch, :]
        smiles_list = batch_df['cano_smiles']
        y_target = torch.cuda.LongTensor(batch_df[task].values)
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list, feature_dicts)
        x_atom = torch.Tensor(x_atom).to(device)
        x_bonds = torch.Tensor(x_bonds).to(device)
        x_atom_index = torch.LongTensor(x_atom_index).to(device)
        x_bond_index = torch.LongTensor(x_bond_index).to(device)
        x_mask = torch.Tensor(x_mask).to(device)
        
        y_pred_ = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
        y_target = y_target.cpu().detach().numpy()
        validInds = np.where((y_target == 0) | (y_target == 1))[0]
        y = np.array([y_target[v] for v in validInds]).astype(float)
        validInds = torch.cuda.LongTensor(validInds).squeeze().to(y_pred_.device)
        y_pred = torch.index_select(y_pred_, 0, validInds)    


        pred_ = F.softmax(y_pred, dim=-1).data.cpu().numpy()[:, 1]
        target_ = y_target
        loss = loss_function(y_pred, torch.cuda.LongTensor(y).to(y_pred_.device))
        losses_list.append(loss.cpu().detach().numpy())
        try:
            y_val_list.extend(target_)
            y_pred_list.extend(pred_)
        except:
            y_val_list = []
            y_pred_list = []
            y_val_list.extend(target_)
            y_pred_list.extend(pred_)
    acc = accuracy(y_val_list, y_pred_list)
    pre_score = precision(y_val_list, y_pred_list)
    recall_score = recall(y_val_list, y_pred_list)
    mcc_score = mcc(y_val_list, y_pred_list)
    roc_score = roc(y_val_list, y_pred_list)
    f1_score = f1(y_val_list, y_pred_list)
    bacc_score = bacc(y_val_list, y_pred_list)
    eval_loss = np.array(losses_list).mean()

    return acc, pre_score, recall_score, mcc_score, roc_score, f1_score, bacc_score, eval_loss


best_epochs = []
best_mccs = []
best_f1s = []
best_accs = []
best_pres = []
best_recalls = []
best_rocs = []
best_baccs = []

tasks_withoutneg = []
tasks_withoutneg_inquery = []

with open(os.path.join(save_dir,'finetune_train_log.txt'), 'a') as f:
    col_name_list = ["epoch", "Accuracy", "Precision", "Recall", "MCC", "ROAUC", "F1", "Balance_Accuracy", "loss"]
    f.write(','.join([str(r) for r in col_name_list]))
    f.write('\n')
with open(os.path.join(save_dir,'finetune_valid_log.txt'), 'a') as f:
    col_name_list = ["epoch", "Accuracy", "Precision", "Recall", "MCC", "ROAUC", "F1", "Balance_Accuracy", "loss"]
    f.write(','.join([str(r) for r in col_name_list]))
    f.write('\n')


#main func, train eack task 
for k, task in enumerate(tasks_name):
    best_task_mcc = 0
    mcc_epoch = 0
    best_task_f1 = 0
    best_task_acc = 0
    best_task_pre = 0
    best_task_recall = 0
    best_task_roc = 0
    best_task_bacc = 0

    print(f'{k} task_name:\t' + task)
    model.load_state_dict(torch.load(args.meta_model_path))  #best meta-learner or fine-tuned models need further fine-tune
    model.to(device)
    optimizer = optim.Adam(model.parameters(), 0.001)

    support_df_decoys, query_df_decoys, positive_query, negative_query = split_task(task_df_decoys, task, random_seed)
    weight = [(positive_query.shape[0]+negative_query.shape[0])/negative_query.shape[0], \
                     (positive_query.shape[0]+negative_query.shape[0])/positive_query.shape[0]]

    loss_function = nn.CrossEntropyLoss(torch.Tensor(weight).to(device), reduction='mean')


    print('start training')

    train_losses = []
    train_mccs = []
    valid_mccs = []
    list_num_train = []
    best_model = model

    for i in range(num_train):
        train(model, task, dataset=support_df_decoys, optimizer=optimizer, loss_function=loss_function)
        train_acc, train_pre_score, train_recall_score, train_mcc_score, train_roc_score, train_f1_score, train_bacc_score, train_loss = eval(model, task, dataset=support_df_decoys, loss_function=loss_function)
        print('num of update train:\t'+str(i)+'\n'\
             +'train_loss'+':'+str(train_loss)+'\n' \
             +'train_acc'+':'+str(train_acc)+'\n' \
             +'train_pre_score'+':' + str(train_pre_score) + '\n' \
             +'train_recall_score'+':' + str(train_recall_score) + '\n'\
             +'train_mcc_score'+':'+str(train_mcc_score)+'\n'\
             +'train_roc_score'+':'+str(train_roc_score)+'\n'\
             +'train_f1_score'+':'+str(train_f1_score)+'\n'\
              +'train_bacc_score'+':'+str(train_bacc_score)+'\n')
        train_losses.append(train_loss)
        train_mccs.append(train_mcc_score)
        list_num_train.append(i)

        #save train results
        results1 = [i, train_acc, train_pre_score, train_recall_score, train_mcc_score, train_roc_score, train_f1_score, train_bacc_score, train_loss]
        with open(os.path.join(save_dir,'finetune_train_log.txt'), 'a') as f:
            f.write(task)
            f.write(','.join([str(r) for r in results1]))
            f.write('\n')

        # try to delete decoys and only evaluate real data
        try:
            support_df_all, query_df_all, _, _ = split_task(task_df_all, task, random_seed)
            query_list_decoys = query_df_decoys['cano_smiles']
            smiles_list_all = list(support_df_all['cano_smiles']) + list(query_df_all['cano_smiles'])  #all real data
            dup_list = list(set(query_list_decoys) & set(smiles_list_all))  #real data in decoys query
            index_list = []
            for index in range(len(query_list_decoys)):
                if query_df_decoys.cano_smiles[index] in dup_list:
                    index_list.append(index)
            query_df = query_df_decoys.iloc[index_list]    #real data in query df
            query_neg_df = query_df[query_df[task] == 0][['cano_smiles', task]]
            if query_neg_df.shape[0] < 1:
                print('This task has no negative sample in query!')
                tasks_withoutneg_inquery.append(task)
                valid_acc, valid_pre_score, valid_recall_score, valid_mcc_score, valid_roc_score, valid_f1_score, valid_bacc_score, valid_loss = eval(
                    model, task, dataset=query_df_decoys, loss_function=loss_function)
                print('task_name:\t' + task + '\n' \
                      + 'num of update train:\t' + str(i) + '\n' \
                      + 'valid_acc' + ':' + str(valid_acc) + '\n' \
                      + 'valid_pre_score' + ':' + str(valid_pre_score) + '\n' \
                      + 'valid_recall_score' + ':' + str(valid_recall_score) + '\n' \
                      + 'valid_mcc_score' + ':' + str(valid_mcc_score) + '\n' \
                      + 'valid_roc_score' + ':' + str(valid_roc_score) + '\n' \
                      + 'valid_f1_score' + ':' + str(valid_f1_score) + '\n' \
                      + 'valid_bacc_score' + ':' + str(valid_bacc_score) + '\n')

            else:
                query_df = query_df.reset_index(drop=True)  
                valid_acc, valid_pre_score, valid_recall_score, valid_mcc_score, valid_roc_score, valid_f1_score, valid_bacc_score, valid_loss = eval(
                    model, task, dataset=query_df, loss_function=loss_function)
                print('task_name:\t' + task + '\n' \
                      + 'num of update train:\t' + str(i) + '\n' \
                      + 'valid_acc' + ':' + str(valid_acc) + '\n' \
                      + 'valid_pre_score' + ':' + str(valid_pre_score) + '\n' \
                      + 'valid_recall_score' + ':' + str(valid_recall_score) + '\n' \
                      + 'valid_mcc_score' + ':' + str(valid_mcc_score) + '\n' \
                      + 'valid_roc_score' + ':' + str(valid_roc_score) + '\n' \
                      + 'valid_f1_score' + ':' + str(valid_f1_score) + '\n' \
                      + 'valid_bacc_score' + ':' + str(valid_bacc_score) + '\n')

        except:
            print('This task has no negative sample')
            tasks_withoutneg.append(task)
            valid_acc, valid_pre_score, valid_recall_score, valid_mcc_score, valid_roc_score, valid_f1_score, valid_bacc_score, valid_loss = eval(
                model, task, dataset=query_df_decoys, loss_function=loss_function)
            print('task_name:\t' + task + '\n' \
                  + 'num of update train:\t' + str(i) + '\n' \
                  + 'valid_acc' + ':' + str(valid_acc) + '\n' \
                  + 'valid_pre_score' + ':' + str(valid_pre_score) + '\n' \
                  + 'valid_recall_score' + ':' + str(valid_recall_score) + '\n' \
                  + 'valid_mcc_score' + ':' + str(valid_mcc_score) + '\n' \
                  + 'valid_roc_score' + ':' + str(valid_roc_score) + '\n' \
                  + 'valid_f1_score' + ':' + str(valid_f1_score) + '\n' \
                  + 'valid_bacc_score' + ':' + str(valid_bacc_score) + '\n')

        valid_mccs.append(valid_mcc_score)

        if valid_mcc_score > best_task_mcc:
            mcc_epoch = i
            best_task_mcc = valid_mcc_score
            best_task_f1 = valid_f1_score
            best_task_acc = valid_acc
            best_task_pre = valid_pre_score
            best_task_recall = valid_recall_score
            best_task_roc = valid_roc_score
            best_task_bacc = valid_bacc_score
            best_model = copy.deepcopy(model)  #best model in 50 epochs

        #save val results
        results2 = [i, valid_acc, valid_pre_score, valid_recall_score, valid_mcc_score, valid_roc_score, valid_f1_score, valid_bacc_score, valid_loss]
        with open(os.path.join(save_dir,'finetune_valid_log.txt'), 'a') as f:
            f.write(task)
            f.write(','.join([str(r) for r in results2]))
            f.write('\n')

    save_model(model=best_model, save_path=save_dir, task_name=task)
    print(f'finished to save base-learner {k}: '+task)


    print('best_epoch:' + str(mcc_epoch) + '\n' + 'best_mcc:' + str(best_task_mcc)+ '\n' + 'best_f1:' + str(best_task_f1))


    best_mccs.append(best_task_mcc)
    best_epochs.append(mcc_epoch)
    best_f1s.append(best_task_f1)
    best_accs.append(best_task_acc)
    best_pres.append(best_task_pre)
    best_recalls.append(best_task_recall)
    best_rocs.append(best_task_roc)
    best_baccs.append(best_task_bacc)




print('best_epochs:' + str(best_epochs) + '\n'
      + 'best_mccs:' + str(best_mccs) + '\n'
      + 'best_f1s:' + str(best_f1s) + '\n'
      + 'best_accs:' + str(best_accs) + '\n'
      + 'best_pres:' + str(best_pres) + '\n'
      + 'best_recalls:' + str(best_recalls) + '\n'
      + 'best_rocs:' + str(best_rocs) + '\n'
      + 'best_baccs' + str(best_baccs) + '\n')

print('best_mccs_mean:' + str(np.mean(best_mccs)) + '\n'
      + 'best_f1s_mean:' + str(np.mean(best_f1s)) + '\n'
      + 'best_accs_mean:' + str(np.mean(best_accs)) + '\n'
      + 'best_pres_mean:' + str(np.mean(best_pres)) + '\n'
      + 'best_recalls_mean:' + str(np.mean(best_recalls)) + '\n'
      + 'best_rocs_mean:' + str(np.mean(best_rocs)) + '\n'
      + 'best_baccs_mean' + str(np.mean(best_baccs)))


result3 = [best_mccs, best_f1s, best_accs, best_pres, best_recalls, best_rocs, best_baccs]
result3_df = pd.DataFrame(result3,index = ['mcc','F1score','accuracy','precision','recall','auroc','balance_accuracy'],columns = tasks_name)

result3_df['mean'] = result3_df.mean(axis=1)
result3_df.to_csv(os.path.join(save_dir,'result.csv'))





















