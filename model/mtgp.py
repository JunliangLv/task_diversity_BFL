import logging
import numpy as np
import time
import torch
import torch.nn as nn
import os
import copy
import gc
from utils import matrix_inverse, set_optimizer
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class MTGP(nn.Module):
    """
    This class implements the inference for Multi-task Gaussian Process with auxiliary latent variables.
    The main feature is the mean-field variational inference. 
    """
    def __init__(self, client_id, dataset, number_of_tasks, num_latent, device):
        super(MTGP, self).__init__()
        self.number_of_tasks_reg = number_of_tasks[0]
        self.number_of_tasks_cla = number_of_tasks[1]
        self.number_of_tasks = sum(number_of_tasks)
        self.number_of_latent = num_latent
        self.client_id = client_id
        self.device = device
        self.use_basemodel = True if not dataset.startswith('syn') else False
        self.dataset = dataset
        
        
    def set_train_test_data(self, reg_data, cla_data, reg_data_valid, cla_data_valid, reg_data_test, cla_data_test):
        self.reg = reg_data
        self.cla = cla_data
        self.data = reg_data + cla_data

        self.reg_valid = reg_data_valid
        self.cla_vliad = cla_data_valid
        self.valid_data = reg_data_valid + cla_data_valid

        self.reg_test = reg_data_test
        self.cla_test = cla_data_test
        self.test_data = reg_data_test + cla_data_test
    
    def set_optimizer(self, optimizer, lr_basemodel, lr_weight, lr_theta):
        self.optimizer = optimizer
        self.lr_basemodel = lr_basemodel
        self.lr_weight = lr_weight
        self.lr_theta = lr_theta

    def set_personalization(self, basemodel_agg, weight_agg, theta_agg, noisereg_agg):
        self.basemodel_agg = basemodel_agg
        self.weight_agg = weight_agg
        self.theta_agg = theta_agg
        self.noisereg_agg = noisereg_agg
#########################################################################################################

    'tool function'
    def metrics_reg(self, y_reg, mean_f_reg): 
        sse = []
        for i in range(len(y_reg)):
            sse.append(torch.sum((y_reg[i] - mean_f_reg[i])**2).item())
        return sse

    def metrics_cla(self, y_cla, mean_f_cla):
        correct = []
        for i in range(len(y_cla)):
            pred = (torch.sigmoid(mean_f_cla[i])>0.5).long()
            pred[pred == 0] = -1
            correct.append(torch.sum(pred == y_cla[i]).item())
        return correct

    @staticmethod
    def rbf_kernel(theta, x1, x2):
        assert x1.shape[1]==x2.shape[1],'dimension does not match'
        theta0 = theta[0]
        theta1 = theta[1]
        N1=x1.shape[0]
        N2=x2.shape[0]
        D=x1.shape[1]
        a=x1.reshape(N1,1,D)
        b=x2.reshape(1,N2,D)
        return theta0*torch.exp(-theta1/2*torch.sum((a-b)**2,dim=2))
 
    def rbf_kernel_mo_hete(self, W, theta, x1, x2=None):
        assert W.shape[1] == len(theta), 'dimension of latent functions does not match'
        assert W.shape[0] == len(x1), 'number of tasks does not match'
        if x2 != None:
            assert W.shape[0] == len(x2), 'number of tasks does not match'
            len_x1 = [len(x) for x in x1]
            len_x2 = [len(y) for y in x2]
            Z = len(theta)
            cov_hete = torch.zeros((sum(len_x1), sum(len_x2))).to(self.device)
            for i, x in enumerate(x1):
                for j, y in enumerate(x2):
                    K = torch.zeros(x.shape[0], y.shape[0]).to(self.device)
                    for z in range(Z):
                        K += W[i, z]*W[j, z]*self.rbf_kernel(theta[z], x, y)
                    cov_hete[sum(len_x1[:i]):sum(len_x1[:(i+1)]), sum(len_x2[:j]):sum(len_x2[:(j+1)])] += K
            return cov_hete
        else:
            len_x1 = [len(x) for x in x1]
            Z = len(theta)
            cov_hete = torch.zeros((sum(len_x1), sum(len_x1))) + torch.eye(sum(len_x1))*1e-3
            cov_hete = cov_hete.to(self.device)
            for i, x in enumerate(x1):
                for j, y in enumerate(x1):
                    K = torch.zeros(x.shape[0], y.shape[0]).to(self.device)
                    if i == j:
                        K += (torch.eye(len_x1[i])*1e-3).to(self.device)
                    for z in range(Z):
                        K += W[i, z]*W[j, z]*self.rbf_kernel(theta[z], x, y)
                    cov_hete[sum(len_x1[:i]):sum(len_x1[:(i+1)]), sum(len_x1[:j]):sum(len_x1[:(j+1)])] += K
            return cov_hete

#########################################################################################################
    'Mean-Field Variational Inference'

    def a_c_predict(self, ym_mean, ym_cov, K_MM_inv, K_pre, K_pre_M): 
        k_C = torch.matmul(K_pre_M, K_MM_inv)
        y_pred_mean = torch.matmul(k_C, ym_mean)
        y_pred_cov = K_pre - torch.matmul(k_C,K_pre_M.T) + torch.matmul(torch.matmul(k_C, ym_cov), k_C.T)
        return y_pred_mean, torch.sqrt((torch.diag(y_pred_cov)+y_pred_mean**2).clamp_(0, None)), y_pred_cov

    def MF(self, iters, mean_prior, cov_prior, g_tilde2_prior, y, len_x, noise_reg, init_mean_posterior, init_cov_posterior):        
        mean_cla_list = [[init_mean_posterior[sum(len_x[:self.number_of_tasks_reg+i]): sum(len_x[:self.number_of_tasks_reg+i+1])] \
            for i in range(self.number_of_tasks_cla)]]
        cov_cla_list = [[init_cov_posterior[sum(len_x[:self.number_of_tasks_reg+i]): sum(len_x[:self.number_of_tasks_reg+i+1]), \
            sum(len_x[:self.number_of_tasks_reg+i]): sum(len_x[:self.number_of_tasks_reg+i+1])] for i in range(self.number_of_tasks_cla)]]
        mean_list = []
        cov_list = []
        E_w_cla_list = [[] for i in range(iters)]
        f_tilde_list = [[] for i in range(iters)]
        H_list = []
        v_list = []
        for MF_iteration in range(iters):
            for i in range(self.number_of_tasks_cla):
                f_tilde_list[MF_iteration].append(torch.sqrt(mean_cla_list[MF_iteration][i]**2 + torch.diag(cov_cla_list[MF_iteration][i])\
                    + g_tilde2_prior[sum(len_x[:self.number_of_tasks_reg+i]): sum(len_x[:self.number_of_tasks_reg+i+1])]))
                E_w_cla_list[MF_iteration].append((1/2/f_tilde_list[MF_iteration][i] * torch.tanh(f_tilde_list[MF_iteration][i]/2)).data)
            H_list.append(torch.block_diag(*([torch.eye(len_x[i]).to(self.device)/noise_reg[i] for i in range(self.number_of_tasks_reg)] \
                + [torch.diag(E_w_cla_list[MF_iteration][i]) for i in range(self.number_of_tasks_cla)])))
            v_list.append(torch.cat([y[i] for i in range(self.number_of_tasks_reg)] \
                + [0.5*torch.matmul(torch.diag(1./E_w_cla_list[MF_iteration][i]), y[self.number_of_tasks_reg + i].float()) \
                    for i in range(self.number_of_tasks_cla)]))
            cov_list.append(matrix_inverse(H_list[MF_iteration] + matrix_inverse(cov_prior)))
            mean_list.append(torch.matmul(cov_list[MF_iteration], \
                torch.matmul(H_list[MF_iteration], v_list[MF_iteration]) + torch.matmul(matrix_inverse(cov_prior), mean_prior)))
            mean_cla_list.append([mean_list[MF_iteration][sum(len_x[:self.number_of_tasks_reg+i]): sum(len_x[:self.number_of_tasks_reg+i+1])] \
                                    for i in range(self.number_of_tasks_cla)])
            cov_cla_list.append([cov_list[MF_iteration][sum(len_x[:self.number_of_tasks_reg+i]): sum(len_x[:self.number_of_tasks_reg+i+1]), \
                sum(len_x[:self.number_of_tasks_reg+i]): sum(len_x[:self.number_of_tasks_reg+i+1])] for i in range(self.number_of_tasks_cla)])
        return mean_list[-1], cov_list[-1], E_w_cla_list[-1]

  
    def train(self, num_iters, MF_iters, global_basemodel, global_weight, global_theta, global_noise_reg, alpha, lamda):
        if self.basemodel_agg:
            self.global_basemodel = global_basemodel.to(self.device)
        if self.weight_agg:
            self.global_weight = global_weight.to(self.device)
        if self.theta_agg:
            self.global_theta = global_theta.to(self.device)
        if self.noisereg_agg:
            self.global_noise_reg = global_noise_reg.to(self.device)
        if self.dataset.startswith('syn'):
            self.global_weight.requires_grad = True
            self.global_theta.requires_grad = True
        else:
            self.global_weight.requires_grad = False
            self.global_theta.requires_grad = False            
        
        
        for iteration in range(num_iters):
            start_time = time.time()
            loss = 0
            mse_reg = np.zeros(self.number_of_tasks_reg)
            N_reg_train = np.zeros(self.number_of_tasks_reg)
            acc_cla = np.zeros(self.number_of_tasks_cla)
            N_cla_train = np.zeros(self.number_of_tasks_cla)
            for _, batch in enumerate(zip(*self.data)):
                # feature extraction via basemodel
                x_reg, len_x_reg, y_reg = [], [], []
                x_cla, len_x_cla, y_cla = [], [], []
                for i in range(self.number_of_tasks_reg):
                    x_reg.append(self.global_basemodel(batch[i][0].to(self.device)))
                    len_x_reg.append(len(batch[i][0]))
                    y_reg.append(batch[i][1].to(self.device).reshape(-1))
                for i in range(self.number_of_tasks_cla):
                    x_cla.append(self.global_basemodel(batch[i + self.number_of_tasks_reg][0].to(self.device)))
                    len_x_cla.append(len(batch[i + self.number_of_tasks_reg][0]))
                    y_cla.append(batch[i + self.number_of_tasks_reg][1].to(self.device).reshape(-1))
                
                global_mean = torch.zeros(sum(len_x_reg)+sum(len_x_cla)).to(self.device)
                global_cov = self.rbf_kernel_mo_hete(self.global_weight, self.global_theta, x_reg+x_cla)                
                global_g_tilde2 = global_mean**2 + torch.diag(global_cov)
                init_local_mean = torch.zeros(sum(len_x_reg)+sum(len_x_cla)).to(self.device)
                init_local_cov = torch.eye(sum(len_x_reg)+sum(len_x_cla)).to(self.device)                
                local_mean, local_cov, E_w_cla = self.MF(MF_iters, global_mean, global_cov, global_g_tilde2, y_reg+y_cla, len_x_reg+len_x_cla, self.global_noise_reg, init_local_mean, init_local_cov)
                
                local_g_tilde2 = local_mean**2 + torch.diag(local_cov)
                mean_train_reg_list = []
                cov_train_reg_list = []
                for i in range(self.number_of_tasks_reg):
                    mean_train_reg_list.append(local_mean[sum(len_x_reg[:i]): sum(len_x_reg[:(i+1)])])
                    cov_train_reg_list.append(local_cov[sum(len_x_reg[:i]): sum(len_x_reg[:(i+1)]), sum(len_x_reg[:i]): sum(len_x_reg[:(i+1)])])
                    self.global_noise_reg[i] = (torch.sum(y_reg[i].ravel()**2 \
                        -2*y_reg[i].ravel()*local_mean[sum(len_x_reg[:i]): sum(len_x_reg[:(i+1)])] \
                            +local_g_tilde2[sum(len_x_reg[:i]): sum(len_x_reg[:(i+1)])])/len(y_reg[i])).data.clamp(1e-2, None)
                
                mean_train_cla_list = []
                cov_train_cla_list = []
                for i in range(self.number_of_tasks_cla):
                    mean_train_cla_list.append(local_mean[sum(len_x_reg)+sum(len_x_cla[:i]): sum(len_x_reg)+sum(len_x_cla[:(i+1)])])
                    cov_train_cla_list.append(local_cov[sum(len_x_reg)+sum(len_x_cla[:i]): sum(len_x_reg)+sum(len_x_cla[:(i+1)]), \
                        sum(len_x_reg)+sum(len_x_cla[:i]): sum(len_x_reg)+sum(len_x_cla[:(i+1)])])

                # update hyperparametes and basemodel
                Elogp_r = 0
                for i in range(self.number_of_tasks_reg):
                    Elogp_r += (2 * torch.dot(y_reg[i], mean_train_reg_list[i]) - torch.dot(mean_train_reg_list[i], mean_train_reg_list[i]) - \
                        torch.trace(cov_train_reg_list[i]) - torch.dot(y_reg[i], y_reg[i]))/(2 * self.global_noise_reg[i]) - len(y_reg[i])*np.log(2 * np.pi)/2 - \
                            len(y_reg[i])*torch.log(self.global_noise_reg[i])/2
                Elogp_c = 0
                for i in range(self.number_of_tasks_cla):
                    Elogp_c += (torch.dot(y_cla[i].float(), mean_train_cla_list[i]) - \
                        torch.matmul(torch.matmul(mean_train_cla_list[i], torch.diag(E_w_cla[i])), mean_train_cla_list[i]) - \
                            torch.trace(torch.matmul(torch.diag(E_w_cla[i]), cov_train_cla_list[i])))/2 - \
                                len(y_cla[i])*np.log(2)

                global_cov_inverse = matrix_inverse(global_cov)
                L = torch.linalg.slogdet(global_cov)[1] - torch.linalg.slogdet(local_cov)[1] \
                    + torch.trace(torch.matmul(global_cov_inverse, local_cov)) \
                        + torch.matmul(torch.matmul(local_mean, global_cov_inverse),local_mean) \
                            -2 * torch.matmul(torch.matmul(local_mean, global_cov_inverse),global_mean)\
                                + torch.matmul(torch.matmul(global_mean, global_cov_inverse),global_mean)\
                                    - sum(len_x_reg + len_x_cla)

                Lw = 0
                for i in range(self.number_of_tasks_cla):
                    local_g_tilde_i = torch.sqrt(local_g_tilde2[sum(len_x_reg)+sum(len_x_cla[:i]): sum(len_x_reg)+sum(len_x_cla[:(i+1)])])
                    global_g_tilde_i = torch.sqrt(global_g_tilde2[sum(len_x_reg)+sum(len_x_cla[:i]): sum(len_x_reg)+sum(len_x_cla[:(i+1)])])
                    Lw += torch.sum(torch.log(torch.cosh(local_g_tilde_i/2))) - torch.sum(torch.log(torch.cosh(global_g_tilde_i/2))) \
                            + torch.dot(global_g_tilde_i**2/2, E_w_cla[i]) - torch.dot(local_g_tilde_i**2/2, E_w_cla[i])
                # logging.info(f'Elogp_r = {Elogp_r.item():.4}, Elogp_c = {Elogp_c.item():.4}, L = {L.item():.4}')                
                elbo = lamda*(L/2 + Lw) - alpha*Elogp_r - Elogp_c
                optimizer = set_optimizer(self.optimizer, self.global_basemodel, self.global_weight, self.global_theta, self.lr_basemodel, \
                    self.lr_weight, self.lr_theta, self.use_basemodel)
                optimizer.zero_grad()
                elbo.backward()
                optimizer.step()
                self.global_theta.data.clamp_(1e-5, None)
                loss += elbo.detach().item()
                
                sse_train, correct_train = self.metrics(y_reg, mean_train_reg_list, y_cla, mean_train_cla_list)    
                           
                # asssessment the model
                if self.number_of_tasks_reg:
                    mse_reg = mse_reg + np.array(sse_train)
                    N_reg_train = N_reg_train + np.array(len_x_reg)
                if self.number_of_tasks_cla:
                    acc_cla = acc_cla + np.array(correct_train)
                    N_cla_train = N_cla_train + np.array(len_x_cla)
            if self.number_of_tasks_reg:
                mse_reg = mse_reg / N_reg_train
            if self.number_of_tasks_cla:
                acc_cla = acc_cla / N_cla_train
            
            logging.info(f"client = {self.client_id}, iter = {iteration}: training: loss = {loss:.4f}, mse = {np.round(mse_reg, 4)}, acc = {np.round(acc_cla, 4)}, time={time.time() - start_time:.4f}")

            gc.collect()
            torch.cuda.empty_cache()
        return self.global_basemodel.cpu(), self.global_weight.cpu(), self.global_theta.cpu(), self.global_noise_reg.cpu()

    def metrics(self, y_reg, mean_f_reg_list, y_cla, mean_f_cla_list):
        sse = self.metrics_reg(y_reg, mean_f_reg_list)
        correct = self.metrics_cla(y_cla, mean_f_cla_list)
        return sse, correct 

    def test(self, MF_iters, global_basemodel, global_weight, global_theta, global_noise_reg, dataset = 'valid', calibration=False):
        if dataset == 'valid':
            data = self.valid_data
        elif dataset == 'test':
            data = self.test_data      
        else:
            raise ValueError("dataset should be in ['valid', 'test']") 
        
        if self.basemodel_agg:
            self.global_basemodel = global_basemodel.to(self.device)
        if self.weight_agg:
            self.global_weight = global_weight.to(self.device)
        if self.theta_agg:
            self.global_theta = global_theta.to(self.device)
        if self.noisereg_agg:
            self.global_noise_reg = global_noise_reg.to(self.device)
        
        if calibration:
            g_list, y_list = [], []
        self.global_basemodel.eval()
        with torch.no_grad():
            train_x, train_y, len_x = [], [], []
            for i in range(self.number_of_tasks_reg):
                train_x.append(self.global_basemodel(self.data[i].dataset.image.to(self.device)))
                train_y.append(self.data[i].dataset.y.to(self.device))
                len_x.append(self.data[i].dataset.__len__())
            for i in range(self.number_of_tasks_cla):
                train_x.append(self.global_basemodel(self.data[self.number_of_tasks_reg + i].dataset.image.to(self.device)))
                train_y.append(self.data[self.number_of_tasks_reg + i].dataset.y.to(self.device))
                len_x.append(self.data[self.number_of_tasks_reg + i].dataset.__len__())

            mu = torch.zeros(sum(len_x)).to(self.device)
            K = self.rbf_kernel_mo_hete(self.global_weight, self.global_theta, train_x)
            init_global_g_tilde2 = torch.zeros(sum(len_x)).to(self.device)
            init_global_mean = torch.zeros(sum(len_x)).to(self.device)
            init_global_cov = torch.eye(sum(len_x)).to(self.device)
            global_mean, global_cov, _ = self.MF(MF_iters, mu, K, init_global_g_tilde2, train_y, len_x, self.global_noise_reg, init_global_mean, init_global_cov)
            global_g_tilde2 = global_mean**2 + torch.diag(global_cov)
            init_local_mean = torch.zeros(sum(len_x)).to(self.device)
            init_local_cov = torch.eye(sum(len_x)).to(self.device)
            local_mean, local_cov, _ = self.MF(MF_iters, global_mean, global_cov, global_g_tilde2, train_y, len_x, self.global_noise_reg, init_local_mean, init_local_cov)

            mse_reg = np.zeros(self.number_of_tasks_reg)
            N_test_reg = np.zeros(self.number_of_tasks_reg)
            acc_cla = np.zeros(self.number_of_tasks_cla)
            N_test_cla = np.zeros(self.number_of_tasks_cla)

            for _, batch_test in enumerate(zip(*data)):
                # feature extraction via basemodel
                x_reg_test_list = []
                y_reg_test_list = [] #
                len_x_reg_test = []
                for i in range(self.number_of_tasks_reg):
                    x_reg_test_list.append(self.global_basemodel(batch_test[i][0].to(self.device)))
                    y_reg_test_list.append(batch_test[i][1].to(self.device))
                    len_x_reg_test.append(len(batch_test[i][0]))
                
                x_cla_test_list = []
                y_cla_test_list = [] #
                len_x_cla_test = []
                for i in range(self.number_of_tasks_cla):
                    x_cla_test_list.append(self.global_basemodel(batch_test[i + self.number_of_tasks_reg][0].to(self.device)))
                    y_cla_test_list.append(batch_test[i + self.number_of_tasks_reg][1].to(self.device))
                    len_x_cla_test.append(len(batch_test[i + self.number_of_tasks_reg][0]))

                K_test = self.rbf_kernel_mo_hete(self.global_weight, self.global_theta, x_reg_test_list+x_cla_test_list)
                K_NM_test = self.rbf_kernel_mo_hete(self.global_weight, self.global_theta, x_reg_test_list+x_cla_test_list, train_x)
                mean_test, _, _ = self.a_c_predict(local_mean, local_cov, matrix_inverse(K), K_test, K_NM_test)
                mean_test_reg_list = [mean_test[sum(len_x_reg_test[:i]): sum(len_x_reg_test[:(i+1)])] for i in range(self.number_of_tasks_reg)]
                mean_test_cla_list = [mean_test[sum(len_x_reg_test) + sum(len_x_cla_test[:i]): sum(len_x_reg_test)+sum(len_x_cla_test[:(i+1)])] for i in range(self.number_of_tasks_cla)]
                sse, correct = self.metrics(y_reg_test_list, mean_test_reg_list, y_cla_test_list, mean_test_cla_list)
                if self.number_of_tasks_reg:
                    mse_reg = mse_reg + np.array(sse)
                    N_test_reg = N_test_reg + np.array(len_x_reg_test)
                if self.number_of_tasks_cla:
                    acc_cla = acc_cla + np.array(correct)
                    N_test_cla = N_test_cla + np.array(len_x_cla_test)
                if calibration:
                    g_list.append(mean_test_cla_list)
                    y_list.append(y_cla_test_list)
            if self.number_of_tasks_reg:
                mse_reg = mse_reg / N_test_reg
            if self.number_of_tasks_cla:
                acc_cla = acc_cla / N_test_cla
        if calibration:
            cali = []
            for i in range(self.number_of_tasks_cla):
                cali_i = torch.tensor([])
                for g, y in zip(g_list, y_list):
                    cali_i = torch.concat([cali_i, torch.concat([g[i].reshape(-1, 1), y[i].reshape(-1, 1)], dim=1).detach().cpu()])
                cali.append(cali_i)
        if calibration:
            return mse_reg, acc_cla, cali
        return mse_reg, acc_cla
    
    def ood_test(self, MF_iters, global_basemodel, global_weight, global_theta, global_noise_reg):
        if self.basemodel_agg:
            self.global_basemodel = global_basemodel.to(self.device)
        if self.weight_agg:
            self.global_weight = global_weight.to(self.device)
        if self.theta_agg:
            self.global_theta = global_theta.to(self.device)
        if self.noisereg_agg:
            self.global_noise_reg = global_noise_reg.to(self.device)
        
        data = self.test_data
        self.global_basemodel.eval()
        with torch.no_grad():
            train_x, train_y, len_x = [], [], []
            for i in range(self.number_of_tasks_reg):
                train_x.append(self.global_basemodel(self.data[i].dataset.image.to(self.device)))
                train_y.append(self.data[i].dataset.y.to(self.device))
                len_x.append(self.data[i].dataset.__len__())
            for i in range(self.number_of_tasks_cla):
                train_x.append(self.global_basemodel(self.data[self.number_of_tasks_reg + i].dataset.image.to(self.device)))
                train_y.append(self.data[self.number_of_tasks_reg + i].dataset.y.to(self.device))
                len_x.append(self.data[self.number_of_tasks_reg + i].dataset.__len__())

            mu = torch.zeros(sum(len_x)).to(self.device)
            K = self.rbf_kernel_mo_hete(self.global_weight, self.global_theta, train_x)
            init_global_g_tilde2 = torch.zeros(sum(len_x)).to(self.device)
            init_global_mean = torch.zeros(sum(len_x)).to(self.device)
            init_global_cov = torch.eye(sum(len_x)).to(self.device)
            global_mean, global_cov, _ = self.MF(MF_iters, mu, K, init_global_g_tilde2, train_y, len_x, self.global_noise_reg, init_global_mean, init_global_cov)
            global_g_tilde2 = global_mean**2 + torch.diag(global_cov)
            init_local_mean = torch.zeros(sum(len_x)).to(self.device)
            init_local_cov = torch.eye(sum(len_x)).to(self.device)
            local_mean, local_cov, _ = self.MF(MF_iters, global_mean, global_cov, global_g_tilde2, train_y, len_x, self.global_noise_reg, init_local_mean, init_local_cov)
            x_reg, g_reg_mean, g_reg_var = [], [], []
            x_cla, g_cla_mean, g_cla_var = [], [], []
            

            for _, batch_test in enumerate(zip(*data)):
                # feature extraction via basemodel
                x_reg_test_list = []
                len_x_reg_test = []
                image_reg_test_list = []
                for i in range(self.number_of_tasks_reg):
                    image_reg_test_list.append(batch_test[i])
                    x_reg_test_list.append(self.global_basemodel(batch_test[i].to(self.device)))
                    len_x_reg_test.append(len(batch_test[i]))
                
                x_cla_test_list = []
                len_x_cla_test = []
                image_cla_test_list = []
                for i in range(self.number_of_tasks_cla):
                    image_cla_test_list.append(batch_test[i + self.number_of_tasks_reg])
                    x_cla_test_list.append(self.global_basemodel(batch_test[i + self.number_of_tasks_reg].to(self.device)))
                    len_x_cla_test.append(len(batch_test[i + self.number_of_tasks_reg]))
                K_test = self.rbf_kernel_mo_hete(self.global_weight, self.global_theta, x_reg_test_list+x_cla_test_list)
                K_NM_test = self.rbf_kernel_mo_hete(self.global_weight, self.global_theta, x_reg_test_list+x_cla_test_list, train_x)
                mean_test, _, cov_test = self.a_c_predict(local_mean, local_cov, matrix_inverse(K), K_test, K_NM_test)
                mean_test_reg_list = [mean_test[sum(len_x_reg_test[:i]): sum(len_x_reg_test[:(i+1)])] for i in range(self.number_of_tasks_reg)]
                mean_test_cla_list = [mean_test[sum(len_x_reg_test) + sum(len_x_cla_test[:i]): sum(len_x_reg_test)+sum(len_x_cla_test[:(i+1)])] for i in range(self.number_of_tasks_cla)]
                cov_test_reg_list = [cov_test[sum(len_x_reg_test[:i]): sum(len_x_reg_test[:(i+1)]), sum(len_x_reg_test[:i]): sum(len_x_reg_test[:(i+1)])] \
                                    for i in range(self.number_of_tasks_reg)]
                cov_test_cla_list = [cov_test[sum(len_x_reg_test) + sum(len_x_cla_test[:i]): sum(len_x_reg_test)+sum(len_x_cla_test[:(i+1)]), \
                                            sum(len_x_reg_test) + sum(len_x_cla_test[:i]): sum(len_x_reg_test)+sum(len_x_cla_test[:(i+1)])] \
                                                for i in range(self.number_of_tasks_cla)]

                x_reg.append(image_reg_test_list)
                g_reg_mean.append(mean_test_reg_list)
                g_reg_var.append([torch.diag(cov) for cov in cov_test_reg_list])
                x_cla.append(image_cla_test_list)
                g_cla_mean.append(mean_test_cla_list)
                g_cla_var.append([torch.diag(cov) for cov in cov_test_cla_list])

            plot = []
            for i in range(self.number_of_tasks_reg):
                plot_result = torch.tensor([])
                image_list = torch.tensor([])
                for x, mean, var in zip(x_reg, g_reg_mean, g_reg_var):
                    image_list = torch.concat([image_list, x[i]])
                    plot_result = torch.concat([plot_result, torch.concat([mean[i].reshape(-1, 1).detach().cpu(), var[i].reshape(-1, 1).detach().cpu()], dim=1)])
                plot.append({"image": image_list, "predict": plot_result})
            for i in range(self.number_of_tasks_cla):
                plot_result = torch.tensor([])
                image_list = torch.tensor([])
                for x, mean, var in zip(x_cla, g_cla_mean, g_cla_var):
                    image_list = torch.concat([image_list, x[i]])
                    plot_result = torch.concat([plot_result, torch.concat([mean[i].reshape(-1, 1).detach().cpu(), var[i].reshape(-1, 1).detach().cpu()], dim=1)])
                plot.append({"image": image_list, "predict": plot_result})            
        return plot