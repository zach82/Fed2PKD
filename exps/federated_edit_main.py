#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path
import pandas as pd
import pickle
import csv
lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt, g_test_inference_new_het_lt, GlobalRepGenerator, getGlobalFeatures, draw_tSNE_Graph, draw_tSNE_Graph_v2
from models import CNNMnistClasses, CNNFemnistClasses
from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem, agg_func_edit, proto_aggregation_edit, agg_func_v1, proto_aggregation_v1, init_nets

"""Heterogeneous Model Setting"""
Nets_Name_List = ['ResNet152', 'ResNet50', 'ResNet101','ResNet152', 'ResNet50', 'ResNet101','ResNet152', 'ResNet50', 'ResNet101','ResNet152', 'ResNet152', 'ResNet50', 'ResNet101','ResNet152', 'ResNet50', 'ResNet101','ResNet152', 'ResNet50', 'ResNet101','ResNet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, g_test_dataset, g_user_groups, g_user_groups_lt, g_classes_list, g_classes_list_gt):
    global_protos = []
    idxs_users = np.arange(args.num_users)
    res_train_acc_list, res_train_loss1_list, res_train_loss2_list, res_train_loss3_list, res_train_total_loss_list = [], [], [], [], []
    res_test_acc_list, res_test_protos_acc_list, g_res_test_acc_list, g_res_test_protos_acc_list, g_res_test_protos_m_acc_list = [], [], [], [], []

    t_local_model_list = local_model_list.copy()

    acc_list, protos_acc_list = test_inference_new_het_lt(args, t_local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    global_test_acc = np.mean(acc_list)
    global_protos_test_acc = np.mean(protos_acc_list)
    
    print('For all users, mean of test acc is {:.5f}'.format(global_test_acc))

    res_test_acc_list.append(global_test_acc)
    res_test_protos_acc_list.append(global_protos_test_acc)

    g_acc_list, g_protos_acc_list, g_protos_m_acc_list = g_test_inference_new_het_lt(args, t_local_model_list, g_test_dataset, g_classes_list, g_user_groups_lt, global_protos)
    g_global_test_acc = np.mean(g_acc_list)
    g_global_protos_test_acc = np.mean(g_protos_acc_list)
    g_global_protos_m_test_acc = np.mean(g_protos_m_acc_list)

    print('For all users, mean of global test acc is {:.5f}'.format(g_global_protos_test_acc))

    g_res_test_acc_list.append(g_global_test_acc)
    g_res_test_protos_acc_list.append(g_global_protos_test_acc)
    g_res_test_protos_m_acc_list.append(g_global_protos_m_test_acc)

    for round in tqdm(range(args.rounds)):
        local_k = []
        local_weights, local_losses, t_local_protos = [], [], {}
        global_train_acc_list, global_train_loss1_list, global_train_loss2_list, global_train_loss3_list, global_train_total_loss_list = [], [], [], [], []
        Generator = GlobalRepGenerator(args, train_dataset)
        loader = Generator.dataloader
        s_images, lab = next(iter(loader))
        s_images = s_images.to(args.device)
        g_local_model_list = local_model_list
        global_features = getGlobalFeatures(args, g_local_model_list, s_images.clone().detach(), global_protos)
        if len(global_protos) > 0:
            for key in global_protos.keys():
                global_protos[key][0].detach()
        print(f'\n | Global Training Round : {round + 1} |\n')
        for idx in idxs_users:
            local_updater = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_updater.update_weights_het_v3(args, idx, global_protos, s_images.clone().detach(), global_features.clone().detach(), model=local_model_list[idx], global_round=round)
            for [label, proto_list] in protos.items():
                for proto in proto_list:
                    if label in t_local_protos:
                        t_local_protos[label].append(proto.detach())
                    else:
                        t_local_protos[label] = [proto.detach()]
            local_weights.append(copy.deepcopy(w))
            global_train_acc_list.append(acc)
            global_train_loss1_list.append(loss['1'])
            global_train_loss2_list.append(loss['2'])
            global_train_loss3_list.append(loss['3'])
            global_train_total_loss_list.append(loss['total'])
            torch.cuda.empty_cache()

        global_train_acc = np.mean(global_train_acc_list)
        global_train_loss1 = np.mean(global_train_loss1_list)
        global_train_loss2 = np.mean(global_train_loss2_list)
        global_train_loss3 = np.mean(global_train_loss3_list)
        global_train_total_loss = np.mean(global_train_total_loss_list)
        print('For all users, mean of train acc is {:.5f}'.format(global_train_acc))
        print('For all users, mean of train loss1 is {:.5f}'.format(global_train_loss1))
        print('For all users, mean of train loss2 is {:.5f}'.format(global_train_loss2))
        print('For all users, mean of train loss3 is {:.5f}'.format(global_train_loss3))
        print('For all users, mean of train loss is {:.5f}'.format(global_train_total_loss))
        res_train_acc_list.append(global_train_acc)
        res_train_loss1_list.append(global_train_loss1)
        res_train_loss2_list.append(global_train_loss2)
        res_train_loss3_list.append(global_train_loss3)
        res_train_total_loss_list.append(global_train_total_loss)
        local_weights_list = local_weights
        for idx in idxs_users:
            # local_model = copy.deepcopy(local_model_list[idx])
            # local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx].load_state_dict(local_weights_list[idx], strict=True)

        t_local_model_list = local_model_list

        for [label, proto_list] in t_local_protos.items():
            stacked_tensor = torch.stack(proto_list)
            t_local_protos[label] = [torch.mean(stacked_tensor.detach(), dim=0)]

        global_protos = t_local_protos.copy()

        for key in global_protos.keys():
            global_protos[key][0] = global_protos[key][0].detach()

        acc_list, protos_acc_list = test_inference_new_het_lt(args, t_local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
        global_test_acc = np.mean(acc_list)
        global_protos_test_acc = np.mean(protos_acc_list)

        print('For all users, mean of test acc is {:.5f}'.format(global_test_acc))
        res_test_acc_list.append(global_test_acc)
        res_test_protos_acc_list.append(global_protos_test_acc)

        g_acc_list, g_protos_acc_list, g_protos_m_acc_list = g_test_inference_new_het_lt(args, t_local_model_list, g_test_dataset, g_classes_list, g_user_groups_lt, global_protos)
        g_global_test_acc = np.mean(g_acc_list)
        g_global_protos_test_acc = np.mean(g_protos_acc_list)
        g_global_protos_m_test_acc = np.mean(g_protos_m_acc_list)

        print('For all users, mean of global test acc is {:.5f}'.format(g_global_protos_test_acc))
        g_res_test_acc_list.append(g_global_test_acc)
        g_res_test_protos_acc_list.append(g_global_protos_test_acc)
        g_res_test_protos_m_acc_list.append(g_global_protos_m_test_acc)
    
    train_frame = pd.DataFrame({'Train Acc' : res_train_acc_list, 'Train Loss1' : res_train_loss1_list, 'Train Loss2' : res_train_loss2_list, 'Train Loss3' : res_train_loss3_list, 'Train Loss' : res_train_total_loss_list})
    test_frame = pd.DataFrame({'Test Acc1' : res_test_acc_list, 'Test Acc2' : res_test_protos_acc_list, 'Global Test Acc1' : g_res_test_acc_list, 'Global Test Acc2' : g_res_test_protos_acc_list})
    file1_n = './Train Res ' + str(args.algo) + str(args.dataset) + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.rounds) + 'r' + str(args.num_users) + 'u' + '.csv'
    file2_n = './Test Res ' + str(args.algo) + str(args.dataset) + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.rounds) + 'r' + str(args.num_users) + 'u' + '.csv'
    train_frame.to_csv(file1_n, sep=',')
    test_frame.to_csv(file2_n, sep=',')

    feature_dict_1 = draw_tSNE_Graph(args, local_model_list, test_dataset, user_groups_lt)
    feature_dict_2 = draw_tSNE_Graph_v2(args, local_model_list, test_dataset, user_groups_lt)

    with open('feature_dict_v1.pkl', 'wb') as f:
        pickle.dump(feature_dict_1, f)
    with open('feature_dict_v2.pkl', 'wb') as f:
        pickle.dump(feature_dict_2, f)

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    g_n_list = np.full((args.num_users,), args.num_classes)
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
        g_k_list = np.full((args.num_users,), args.shots)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
        g_k_list = np.full((args.num_users,), args.shots)
    elif args.dataset =='cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
        g_k_list = np.full((args.num_users,), args.shots)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
        g_k_list = np.full((args.num_users,), args.shots)
    
    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

    g_train_dataset, g_test_dataset, g_user_groups, g_user_groups_lt, g_classes_list, g_classes_list_gt = get_dataset(args, g_n_list, g_k_list)

    local_model_list = []
    if args.dataset == 'cifar100' or args.dataset == 'cifar10':
        local_model_list = init_nets(n_parties=args.num_users,nets_name_list=Nets_Name_List,num=args.num_classes)
        for i in range(args.num_users):
            net_name = Nets_Name_List[i]
            local_model_list[i].to(args.device)
            local_model_list[i].train()
    else:
        for i in range(args.num_users):
            if args.dataset == 'mnist':
                if args.mode == 'model_heter':
                    args.out_channels = 20
                else:
                    args.out_channels = 20
                random_class = random.choice(CNNMnistClasses)
                local_model = random_class(args=args)

            elif args.dataset == 'femnist':
                if args.mode == 'model_heter':
                    args.out_channels = 20
                else:
                    args.out_channels = 20
                random_class = random.choice(CNNFemnistClasses)
                local_model = random_class(args=args)

            local_model.to(args.device)
            local_model.train()
            local_model_list.append(local_model)

    FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, g_test_dataset, g_user_groups, g_user_groups_lt, g_classes_list, g_classes_list_gt)