import copy
import json
import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import List
from mtgp import MTGP
from basemodel import synmodel1d
from utils import set_seed
from preprocess import syn_dataset

def eval_model(GPs, super_param, args):

    weight, theta, noise_reg = super_param

    plot_res = dict()

    for id in range(args.num_client):
        plot_client = GPs[id].ood_test(args.MF_iters, synmodel1d(), copy.deepcopy(weight), copy.deepcopy(theta), copy.deepcopy(noise_reg))
        plot_res[id] = plot_client
    return plot_res

def warming_up(clients, args):
    '''
    warming up before training model, including loading global model in each client, 
    running training process for some interations.
    '''
    weight = torch.tensor(args.init_task_weight, dtype=torch.float32)
    theta = torch.tensor(args.init_RBF, dtype=torch.float32)
    noise_reg = torch.tensor(args.init_noisereg, dtype=torch.float32)
    logging.info("-------------warming up-------------")
    GPs = list()
    agg_weight = torch.zeros_like(weight)
    agg_theta = torch.zeros_like(theta)
    agg_noisereg = torch.zeros_like(noise_reg)

    for id in range(args.num_client):
        model = MTGP(id, args.dataset, args.num_task, args.num_latent, args.device[id])
        model.set_train_test_data(clients[id]['reg'], clients[id]['cla'], [], [], clients[id]['reg_test'], clients[id]['cla_test'])
        model.set_optimizer(args.optimizer, 0.0, args.lr_weight, args.lr_theta)
        model.set_personalization(basemodel_agg=True, weight_agg=True, theta_agg=True, noisereg_agg=True)
        _, global_weight, global_theta, global_noise_reg = model.train(args.warm_up, args.MF_iters, \
            synmodel1d(), copy.deepcopy(weight), copy.deepcopy(theta), copy.deepcopy(noise_reg), args.alpha, args.lamda)

        agg_weight += global_weight.data
        agg_theta += global_theta.data
        agg_noisereg += global_noise_reg.data
        GPs.append(model)
    weight = agg_weight/args.num_client
    theta = agg_theta/args.num_client
    
    logging.info("-------------warming up finished--------------")
    return GPs, [weight, theta, noise_reg]

def train_model(GPs, super_param, args):
    logging.info("---------------Training---------------")
    chosen = np.zeros(args.num_client)
    weight, theta, noise_reg = super_param
    for epoch in range(args.epoches):
        client_ids = np.random.choice(np.arange(args.num_client), args.num_agg_client, replace=False)
        agg_weight = torch.zeros_like(weight)
        agg_theta = torch.zeros_like(theta)
        agg_noisereg = torch.zeros_like(noise_reg)

        for id in client_ids:
            chosen[id] += 1
            _, global_weight, global_theta, global_noise_reg = GPs[id].train(args.inner_epoches, args.MF_iters, \
                synmodel1d(), copy.deepcopy(weight), copy.deepcopy(theta), copy.deepcopy(noise_reg), args.alpha, args.lamda)
            agg_weight += global_weight.data
            agg_theta += global_theta.data
            agg_noisereg += global_noise_reg.data
        weight = agg_weight/args.num_agg_client
        theta = agg_theta/args.num_agg_client
        noise_reg = agg_noisereg/args.num_agg_client

    return GPs, [weight, theta, noise_reg]



def main():
    #### set parameters ####
    set_seed(123)
    parse = argparse.ArgumentParser()

    parse.add_argument("--dataset", type=str, default="syn_f1")

    parse.add_argument("--epoches", type=int, default=20, help="epoches of the whole train process")
    parse.add_argument("--inner_epoches", type=int, default=2, help="epoches of inner train process of each client in one epoch")
    parse.add_argument("--warm_up", type=int, default=2, help='epoches of pretrain for each client')
    parse.add_argument("--MF_iters", type=int, default=2, help="the number of iterations of mean field")
    parse.add_argument("--device_id", type=int, default=0, help='device id in gpu training')
    
    parse.add_argument("--num_agg_client", type=int, default=5, help="the number of aggregated clients in one epoch")
    parse.add_argument("--alpha", type=float, default=0.5)
    parse.add_argument("--lamda", type=float, default=0.1)

    parse.add_argument("--batch_size",type=int, default=100, help="batch size of dataloader")

    parse.add_argument("--num_latent", type=int, default=2, help="the number of latent functions in GP")
    
    parse.add_argument("--optimizer", default="AdamW", choices=["Adam","AdamW","SGD"], help="optimizer for training")
    parse.add_argument("--lr_weight", type=float, default=1e-3, help="learning rate for weight of different kernels")
    parse.add_argument("--lr_theta", type=float, default=1e-3, help="learning rate for parameters of RBF kernel")
    parse.add_argument("--lr_decay", type=float, default=1., help="exponential decay factor for learning rate")

    args = parse.parse_args()

    with open(os.path.join("../data/synthetic", args.dataset, 'param.json'), 'r') as f:
        param = json.load(f)
    # fed task
    args.num_client = param['num_client']
    args.num_reg_task = param['num_reg'] #
    args.num_cla_task = param['num_cla'] #
    args.num_task = [param['num_reg'], param['num_cla']] #
    args.trainsize = param['data_size']
    args.reg_trainsize = 50 # param['data_size']
    args.cla_trainsize = 100 # param['data_size']
    args.testrange = param['xrange']
    args.init_task_weight = param['w'] #
    args.init_RBF = param['theta']
    args.init_noisereg = param['sigma2'] #
    args.device = ['cuda:0']*args.num_client

    assert args.num_agg_client <= args.num_client, logging.info("the number of aggregation clients should be less than total number of clients")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info(str(args))

    # client data
    train_data = torch.load(os.path.join("../data/synthetic/", args.dataset, 'data.pth'))
    clients = dict()
    for id in range(args.num_client):
        client_data = {"reg":[], "cla":[], "reg_test":[], "cla_test":[]}
        for i in range(args.num_reg_task):
            reg_dataset = syn_dataset(train_data[id]['x_reg'][i*args.reg_trainsize: (i+1)*args.reg_trainsize], \
                                      train_data[id]['y_reg'][i*args.reg_trainsize: (i+1)*args.reg_trainsize].float())
            reg_loader = DataLoader(reg_dataset, batch_size=args.batch_size, shuffle=True)
            client_data['reg'].append(reg_loader)
            reg_testloader = DataLoader(syn_dataset(torch.linspace(0, args.testrange, 1000).reshape(-1, 1)), shuffle=False, batch_size=args.batch_size)
            client_data['reg_test'].append(reg_testloader)
        for i in range(args.num_cla_task):
            cla_dataset = syn_dataset(train_data[id]['x_cla'][i*args.cla_trainsize: (i+1)*args.cla_trainsize], \
                                      train_data[id]['y_cla'][i*args.cla_trainsize: (i+1)*args.cla_trainsize].long())
            cla_loader = DataLoader(cla_dataset, batch_size=args.batch_size, shuffle=True)
            client_data['cla'].append(cla_loader)
            cla_testloader = DataLoader(syn_dataset(torch.linspace(0, args.testrange, 1000).reshape(-1, 1)), shuffle=False, batch_size=args.batch_size)
            client_data['cla_test'].append(cla_testloader)           
        clients[id] = client_data
    
    GPs, super_params = warming_up(clients, args)
    GPs, super_params = train_model(GPs, super_params, args)
    plot = eval_model(GPs, super_params, args)
    logging.info(f"Aggregate super parameters: {super_params}")
    torch.save(plot, os.path.join("../data/synthetic/", args.dataset, f"simulation.pth"))


    ########### only conduct regression tasks ##########
    args.num_reg_task = 1 #
    args.num_cla_task = 0 #
    args.num_task = [1, 0] #
    args.init_task_weight = [param['w'][0]] #
    args.noise_reg = param['sigma2'] #

    assert args.num_agg_client <= args.num_client, logging.info("the number of aggregation clients should be less than total number of clients")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info(str(args))

    # client data
    clients = dict()
    for id in range(args.num_client):
        client_data = {"reg":[], "cla":[], "reg_test":[], "cla_test":[]}
        for i in range(args.num_reg_task):
            reg_dataset = syn_dataset(train_data[id]['x_reg'][i*args.reg_trainsize: (i+1)*args.reg_trainsize], \
                                      train_data[id]['y_reg'][i*args.reg_trainsize: (i+1)*args.reg_trainsize].float())
            reg_loader = DataLoader(reg_dataset, batch_size=args.batch_size, shuffle=True)
            client_data['reg'].append(reg_loader)
            reg_testloader = DataLoader(syn_dataset(torch.linspace(0, args.testrange, 1000).reshape(-1, 1)), shuffle=False, batch_size=args.batch_size)
            client_data['reg_test'].append(reg_testloader)        
        clients[id] = client_data
    
    GPs, super_params = warming_up(clients, args)
    GPs, super_params = train_model(GPs, super_params, args)
    plot = eval_model(GPs, super_params, args)
    logging.info(f"Aggregate super parameters: {super_params}")
    torch.save(plot, os.path.join("../data/synthetic/", args.dataset, f"simulationreg.pth"))

    ############### only conduct classification tasks #############
    args.num_reg_task = 0
    args.num_cla_task = 1
    args.num_task = [0, 1]
    args.init_task_weight = [param['w'][1]] #
    args.noise_reg = [] #

    assert args.num_agg_client <= args.num_client, logging.info("the number of aggregation clients should be less than total number of clients")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info(str(args))

    # client data
    clients = dict()
    for id in range(args.num_client):
        client_data = {"reg":[], "cla":[], "reg_test":[], "cla_test":[]}
        for i in range(args.num_cla_task):
            cla_dataset = syn_dataset(train_data[id]['x_cla'][i*args.cla_trainsize: (i+1)*args.cla_trainsize], \
                                      train_data[id]['y_cla'][i*args.cla_trainsize: (i+1)*args.cla_trainsize].long())
            cla_loader = DataLoader(cla_dataset, batch_size=args.batch_size, shuffle=True)
            client_data['cla'].append(cla_loader)
            cla_testloader = DataLoader(syn_dataset(torch.linspace(0, args.testrange, 1000).reshape(-1, 1)), shuffle=False, batch_size=args.batch_size)
            client_data['cla_test'].append(cla_testloader)           
        clients[id] = client_data
    
    GPs, super_params = warming_up(clients, args)
    GPs, super_params = train_model(GPs, super_params, args)
    plot = eval_model(GPs, super_params, args)
    logging.info(f"Aggregate super parameters: {super_params}")
    torch.save(plot, os.path.join("../data/synthetic/", args.dataset, f"simulationcla.pth"))

if __name__ == "__main__":
    main()