#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.04,
                        help='the fraction of clients: C')
    parser.add_argument('--train_ep', type=int, default=1,
                        help="the number of local episodes: E")
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--alg', type=str, default='fedproto', help="algorithms")
    parser.add_argument('--mode', type=str, default='model_heter', help="mode")
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    parser.add_argument('--ks', type=int, default=50, help="ks")

    # other arguments
    parser.add_argument('--data_dir', type=str, default='../data/', help="directory of dataset")
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=-1, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--test_ep', type=int, default=10, help="num of test episodes for evaluation")

    # Local arguments
    parser.add_argument('--ways', type=int, default=7, help="num of classes")
    parser.add_argument('--shots', type=int, default=500, help="num of shots")
    parser.add_argument('--train_shots_max', type=int, default=230, help="num of shots")
    parser.add_argument('--test_shots', type=int, default=15, help="num of shots")
    parser.add_argument('--stdev', type=int, default=2, help="stdev of ways")
    parser.add_argument('--ld', type=float, default=0.9, help="weight of proto loss")
    parser.add_argument('--ft_round', type=int, default=10, help="round of fine tuning")
    parser.add_argument('--tau', type=float, default=0.07, help="tau for contrast")
    parser.add_argument('--tau1', type=float, default=0.07, help="tau for contrast")
    parser.add_argument('--algo', type=str, default='mine', help="tau for contrast")
    parser.add_argument('--lag', type=bool, default=True, help="explore the number of labels or not")
    parser.add_argument('--t_global', type=bool, default=False, help="using the protos to test the global data")
    parser.add_argument('--var', type=bool, default=False, help="是否逐渐变化")
    parser.add_argument('--ldc', type=float, default=0.9, help="对比学习正则项")
    parser.add_argument('--k', type=int, default=50, help="对比学习正则项")
    parser.add_argument('--lambda1', type=float, default=0.5, help="公共数据集")
    parser.add_argument('--eta', type=float, default=0.1, help="公共数据集")
    args = parser.parse_args()
    return args
