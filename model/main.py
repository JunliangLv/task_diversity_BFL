import copy
import os
import argparse
import logging
import json
from pickle import TRUE
from random import sample
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from typing import List
from PIL import Image
from mtgp import MTGP
from basemodel import CNNModel

from collections import OrderedDict
from utils import set_seed
from preprocess import sample_celeba, sample_dogcat, load_celeba, load_dogcat, generate_clients
from uncertainty import get_ood_sample, get_ood_density
from preprocess import qmul_dataset


def eval_model(GPs, basemodel, super_params, args, dataset="valid"):
    weight, theta, noise_reg = super_params
    mse_eval = np.array([0.]*args.num_reg_task)
    acc_eval = np.array([0.]*args.num_cla_task)

    for id in range(args.num_client):
        mse_reg, acc_cla = GPs[id].test(args.MF_iters, \
            copy.deepcopy(basemodel), copy.deepcopy(weight), copy.deepcopy(theta), \
                copy.deepcopy(noise_reg), dataset = dataset)
        mse_eval += mse_reg
        acc_eval += acc_cla
    return mse_eval/args.num_client, acc_eval/args.num_client

def warming_up(clients, args):
    if args.dataset == 'celeba' or args.dataset == 'dogcat':
        basemodel = CNNModel(args.num_dim, freeze=args.freeze, mlp=args.mlp)
    else:
        raise ValueError("dataset doesnot exist")
    weight = torch.tensor(args.init_task_weight, dtype=torch.float32)
    theta = torch.tensor(args.init_RBF, dtype=torch.float32)
    noise_reg = torch.tensor(args.init_noisereg, dtype=torch.float32)

    logging.info("-------------warming up-------------")
    GPs = list()
    params = OrderedDict()
    for n, p in basemodel.state_dict().items():
        params[n] = torch.zeros_like(p.data)
    agg_weight = torch.zeros_like(weight)
    agg_theta = torch.zeros_like(theta)
    agg_noisereg = torch.zeros_like(noise_reg)

    for id in range(args.num_client):
        model = MTGP(id, args.dataset, args.num_task, args.num_latent, args.device[id])
        model.set_train_test_data(clients[id]['train_reg'], clients[id]['train_cla'], \
            clients[id]['valid_reg'], clients[id]['valid_cla'], clients[id]['test_reg'], clients[id]['test_cla'])
        model.set_optimizer(args.optimizer, args.lr_basemodel, args.lr_weight, args.lr_theta)
        model.set_personalization(basemodel_agg=True, weight_agg=True, theta_agg=True, noisereg_agg=True)
        global_basemodel, global_weight, global_theta, global_noise_reg = model.train(args.warm_up, args.MF_iters, \
            copy.deepcopy(basemodel), copy.deepcopy(weight), copy.deepcopy(theta), copy.deepcopy(noise_reg), \
                args.alpha, args.lamda)

        for n, p in global_basemodel.state_dict().items():
            params[n] += p.data
        agg_weight += global_weight.data
        agg_theta += global_theta.data
        agg_noisereg += global_noise_reg.data
        GPs.append(model)
    for n, p in params.items():
        params[n] = p / args.num_client
    basemodel.load_state_dict(params)
    weight = agg_weight/args.num_client
    theta = agg_theta/args.num_client
    noise_reg = agg_noisereg/args.num_client
    logging.info("-------------warming up finished--------------")
    mse_eval, acc_eval = eval_model(GPs, basemodel, [weight, theta, noise_reg], args)
    logging.info(f"Validating: mse={[round(mse_st, 4) for mse_st in mse_eval]}, acc={[round(acc_st, 4) for acc_st in acc_eval]}")
    return GPs, basemodel, [weight, theta, noise_reg]

def train_model(GPs, basemodel, super_param, args):
    logging.info("---------------Training---------------")
    chosen = np.zeros(args.num_client)
    weight, theta, noise_reg = super_param
    for id in range(args.num_client):
        GPs[id].set_personalization(basemodel_agg=args.basemodel_agg, weight_agg=args.weight_agg, theta_agg=args.theta_agg, noisereg_agg=args.noisereg_agg)
    for epoch in range(args.epoches):
        client_ids = np.random.choice(np.arange(args.num_client), args.num_agg_client, replace=False)
        params = OrderedDict()
        for n, p in basemodel.state_dict().items():
            params[n] = torch.zeros_like(p.data)
        agg_weight = torch.zeros_like(weight)
        agg_theta = torch.zeros_like(theta)
        agg_noisereg = torch.zeros_like(noise_reg)

        for id in client_ids:
            chosen[id] += 1
            global_basemodel, global_weight, global_theta, global_noise_reg = GPs[id].train(args.inner_epoches, args.MF_iters, \
                copy.deepcopy(basemodel), copy.deepcopy(weight), copy.deepcopy(theta), copy.deepcopy(noise_reg), args.alpha, args.lamda)
            for n, p in global_basemodel.state_dict().items():
                params[n] += p.data
            agg_weight += global_weight.data
            agg_theta += global_theta.data
            agg_noisereg += global_noise_reg.data
        for n, p in params.items():
            params[n] = p / args.num_agg_client
        basemodel.load_state_dict(params)
        weight = agg_weight/args.num_agg_client
        theta = agg_theta/args.num_agg_client
        noise_reg = agg_noisereg/args.num_agg_client

        if epoch%args.valid_per_epoch == 0:
            mse_eval, acc_eval = eval_model(GPs, basemodel, [weight, theta, noise_reg], args, dataset='test')
            logging.info(f"Validating: epoch = {epoch}, mse={[round(mse_st, 4) for mse_st in mse_eval]}, acc={[round(acc_st, 4) for acc_st in acc_eval]}")
    return GPs, basemodel, [weight, theta, noise_reg]

def main():
    set_seed(123)
    parse = argparse.ArgumentParser()

    parse.add_argument("--dataset", type=str, default="celeba")

    parse.add_argument("--epoches", type=int, default=70, help="epoches of the whole train process")
    parse.add_argument("--inner_epoches", type=int, default=2, help="epoches of inner train process of each client in one epoch") #
    parse.add_argument("--warm_up", type=int, default=2, help='epoches of pretrain for each client') #
    parse.add_argument("--MF_iters", type=int, default=2, help="the number of iterations of mean field")
    parse.add_argument("--valid_per_epoch", type=int, default=2, help="how many epoches model is validated after")
    parse.add_argument("--device_id", type=int, nargs='+', default=[0], help='device id in parallel or use first_id in gpu training')

    parse.add_argument("--num_client", type=int, default=10, help="the number of clients")
    parse.add_argument("--num_task", type=int, nargs='+', default=[1, 1], help="[the number of reg tasks, the number of cla tasks]")
    parse.add_argument("--num_agg_client", type=int, default=8, help="the number of aggregated clients in one epoch")
    parse.add_argument("--basemodel_agg", type=bool, default=True)
    ## whether or not to aggregate hyperparameters of base MOGP (ablation study 1)
    parse.add_argument("--weight_agg", action='store_true', help="whether to aggregate weight")
    parse.add_argument("--theta_agg", action='store_true', help='whether to aggregate theta')
    parse.add_argument("--noisereg_agg", action='store_true', help='whether to aggregate regression noise')

    parse.add_argument("--scale", type=bool, default=True, help='whether to scale regression target y')
    parse.add_argument("--downsampling", type=bool, default=False, help="downsampling or not according exponential adjusted proportion")
    parse.add_argument("--reg_task_name", type=str, nargs='+', default=[], help="regression task names")
    parse.add_argument("--cla_task_name", type=str, nargs='+', default=[], help="classification task names")    
    parse.add_argument("--sample_size", type=int, default=500, help="sample size in single task")
    parse.add_argument("--batch_size",type=int, default=32, help="batch size of dataloader")

    parse.add_argument("--freeze", default='none', choices=['none', 'bottom', 'all'], help="freeze the basemodel or not")
    parse.add_argument("--num_latent", type=int, default=2, help="the number of latent functions in GP")
    parse.add_argument("--mlp", action='store_true', help="whether or not to add mlp layer to backbone") # 
    parse.add_argument("--num_dim", type=int, default=100, help="the dimension of output of basemodel")
    parse.add_argument("--init_task_weight", type=str, default="[[0.9, 0.1],[0.1, 0.9]]", help="initial weight for all kernels, must be str") # best 0.9&0.1
    parse.add_argument("--init_RBF", type=str, default="[[1.0, 0.01],[1.0, 0.01]]", help="inital parameters of RBF kernel, must be str") #
    parse.add_argument("--init_noisereg", type=str, default="[0.1]", help='initial gaussian noise') #

    parse.add_argument("--lamda", type=float, default=0.1, help="adjust coeffience for regularization term") #
    parse.add_argument("--alpha", type=float, default=0.5, help='adjust coeffience for regression term') #

    parse.add_argument("--optimizer", default="AdamW", choices=["Adam","AdamW","SGD"], help="optimizer for training")
    parse.add_argument("--lr_basemodel", type=float, default=1e-4, help="learning rate for basemodel")
    parse.add_argument("--lr_weight", type=float, default=1e-3, help="learning rate for weight of different kernels")
    parse.add_argument("--lr_theta", type=float, default=1e-3, help="learning rate for parameters of RBF kernel")

    args = parse.parse_args()
    
    args.init_task_weight = json.loads(args.init_task_weight)
    args.init_RBF = json.loads(args.init_RBF)
    args.init_noisereg = json.loads(args.init_noisereg)

    args.total_num_task = sum(args.num_task)
    num_reg_task = args.num_task[0]
    num_cla_task = args.num_task[1]
    assert args.num_agg_client <= args.num_client, logging.info("the number of aggregation clients shoulb be less than total number of clients")
    assert len(args.reg_task_name) == num_reg_task, logging.info("The number of regression names does not match the number of given tasks")
    assert len(args.cla_task_name) == num_cla_task, logging.info("The number of classification names does not match the number of given tasks")
    assert np.array(args.init_task_weight).shape[0] == args.total_num_task and np.array(args.init_task_weight).shape[1] == args.num_latent, logging.info(f"Wrong initial task weight, should be shape num_task*num_latent {args.total_num_task}*{args.num_latent}")
    assert np.array(args.init_RBF).shape[0] == args.num_latent and np.array(args.init_RBF).shape[1] == 2, logging.info(f"Wrong initial RBF parameters, should be num_latent*2 {args.num_latent}*{2}")
    device = {i: f'cuda:{args.device_id[i%len(args.device_id)]}' for i in range(args.num_client)} if torch.cuda.is_available() \
        else {i: 'cpu' for i in range(args.num_client)}

    args.num_reg_task = num_reg_task
    args.num_cla_task = num_cla_task
    args.device = device
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info(str(args))

    if args.dataset == 'celeba':
        transform = transforms.Compose([
                    transforms.Resize([256]),
                    transforms.CenterCrop([224]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        args.transform = transform
        train, valid, test = load_celeba()
        train_reg_data, train_cla_data = sample_celeba(train, args.reg_task_name, args.cla_task_name, args.sample_size, downsampling=args.downsampling ,alpha=0.1)
        valid_reg_data, valid_cla_data = sample_celeba(valid, args.reg_task_name, args.cla_task_name, 2500, downsampling=args.downsampling, alpha=0.1)
        test_reg_data, test_cla_data = sample_celeba(test, args.reg_task_name, args.cla_task_name, 2500, downsampling=args.downsampling, alpha=0.1)
        clients = generate_clients(train_reg_data, train_cla_data, valid_reg_data, valid_cla_data, test_reg_data, test_cla_data, args)           
    elif args.dataset == 'dogcat':
        train_image, train_label, valid_image, valid_label, test_image, test_label = load_dogcat()
        train_reg_data, train_cla_data = sample_dogcat(train_image, train_label, sample_size=args.sample_size)
        valid_reg_data, valid_cla_data = sample_dogcat(valid_image, valid_label, sample_size=None)
        test_reg_data, test_cla_data = sample_dogcat(test_image, test_label, sample_size=None)
        clients = generate_clients(train_reg_data, train_cla_data, valid_reg_data, valid_cla_data, test_reg_data, test_cla_data, args)     
    else:
        raise ValueError("Dataset is not assigned")


    GPs, basemodel, super_params = warming_up(clients, args)
    GPs, basemodel, super_params = train_model(GPs, basemodel, super_params, args)

    mse_test, acc_test = eval_model(GPs, basemodel, super_params, args, dataset='test')

    logging.info(f"Testing: mse={[round(mse, 4) for mse in mse_test]}, acc={[round(acc, 4) for acc in acc_test]}")

    #### we provide code about ood test and calibration
    # # ood test plot
    # weight, theta, noise_reg = super_params
    # ood_loader = DataLoader(get_ood_sample(dataset=args.dataset, sample_size=18), batch_size=20, shuffle=False)
    # ood_res = {}
    # for id in range(args.num_client):
    #     GPs[id].test_data = [ood_loader, ood_loader]
    #     ood_plot = GPs[id].ood_test(args.MF_iters, copy.deepcopy(basemodel), copy.deepcopy(weight), copy.deepcopy(theta), copy.deepcopy(noise_reg))
    #     ood_res[id] = ood_plot
    # torch.save(ood_res, os.path.join("../data", args.dataset, "ood_plot.pth"))

    # # ood test density
    # weight, theta, noise_reg= super_params
    # trainset, testset, oodset = get_ood_density(args.dataset, num_sample=500)
    # trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    # testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    # oodloader = DataLoader(oodset, batch_size=args.batch_size, shuffle=False)
    # train_res = {}
    # test_res = {}
    # ood_res = {}
    # for id in range(args.num_client):
    #     GPs[id].test_data = [trainloader, trainloader]
    #     train_density = GPs[id].ood_test(args.MF_iters, copy.deepcopy(basemodel), copy.deepcopy(weight), copy.deepcopy(theta), copy.deepcopy(noise_reg))
    #     train_res[id] = train_density
    #     GPs[id].test_data = [testloader, testloader]
    #     test_density = GPs[id].ood_test(args.MF_iters, copy.deepcopy(basemodel), copy.deepcopy(weight), copy.deepcopy(theta), copy.deepcopy(noise_reg))
    #     test_res[id] = test_density
    #     GPs[id].test_data = [oodloader, oodloader]
    #     ood_density = GPs[id].ood_test(args.MF_iters, copy.deepcopy(basemodel), copy.deepcopy(weight), copy.deepcopy(theta), copy.deepcopy(noise_reg))
    #     ood_res[id] = ood_density
    # torch.save(train_res, os.path.join("../data", args.dataset, "train_density.pth"))
    # torch.save(test_res, os.path.join("../data", args.dataset, "test_density.pth"))
    # torch.save(ood_res, os.path.join("../data", args.dataset, "ood_density.pth"))
    
    # calibration
    # weight, theta, noise_reg = super_params
    # cali_res = [torch.tensor([]) for _ in range(args.num_cla_task)]
    # for id in range(args.num_client):
    #     _, _, cali = GPs[id].test(args.MF_iters, copy.deepcopy(basemodel), \
    #         copy.deepcopy(weight), copy.deepcopy(theta), copy.deepcopy(noise_reg), dataset = 'test', calibration=True)
    #     for i in range(args.num_cla_task):
    #         cali_res[i] = torch.concat([cali_res[i], cali[i]])
    # torch.save(cali_res, os.path.join("../data", args.dataset, "calibration.pth"))




if __name__ == "__main__":
    main()