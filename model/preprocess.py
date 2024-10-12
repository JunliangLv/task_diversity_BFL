import os
import numpy as np
import pandas as pd
import re
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

def load_celeba():
    partition = pd.read_csv("../data/celeba/Anno/list_eval_partition.txt", sep=' ', header=None, names=['name', 'partition'])
    train = partition[partition['partition'] == 0]
    valid = partition[partition['partition'] == 1]
    test = partition[partition['partition'] == 2]
    reg = pd.read_csv("../data/celeba/Anno/list_landmarks_align_celeba.csv", index_col=0)
    cla = pd.read_csv("../data/celeba/Anno/list_attr_celeba.csv", index_col=0)

    train = train.join(reg, how='left', on=['name'])
    train = train.join(cla, how='left', on=['name'])

    valid = valid.join(reg, how='left', on=['name'])
    valid = valid.join(cla, how='left', on=['name'])

    test = test.join(reg, how='left', on=['name'])
    test = test.join(cla, how='left', on=['name'])

    return train, valid, test


def load_dogcat():
    def read_traintest_data(datatype, transform):
        path_dog = os.path.join("../data/dogcat", datatype, "dogs")
        path_cat = os.path.join("../data/dogcat", datatype, "cats")
        images = []
        labels_cla = []
        labels_reg = []
        for imgname in os.listdir(path_dog):
            if "dog" in imgname:
                image = Image.open(os.path.join(path_dog, imgname))
                images.append(transform(image))
                labels_cla.append(1)
                labels_reg.append(1. + np.random.randn()*0.5)
        for imgname in os.listdir(path_cat):
            if "cat" in imgname:
                image = Image.open(os.path.join(path_cat, imgname))
                images.append(transform(image))
                labels_cla.append(-1)
                labels_reg.append(-1. + np.random.randn()*0.5)
        return images, np.array(labels_reg), np.array(labels_cla) 
    transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])   
    train_images, train_labels_reg, train_labels_cla = read_traintest_data("train", transform)
    test_images, test_labels_reg, test_labels_cla = read_traintest_data("test", transform)
    # split traintest
    train_id, valid_id = train_test_split(np.arange(len(train_images)), test_size=2000)
    train_images = torch.stack(train_images)
    valid_images = train_images[valid_id]
    valid_labels_reg = train_labels_reg[valid_id]
    valid_labels_cla = train_labels_cla[valid_id]

    train_images = train_images[train_id]
    train_labels_reg = train_labels_reg[train_id]
    train_labels_cla = train_labels_cla[train_id]

    test_images = torch.stack(test_images)
    return train_images, {'reg':train_labels_reg, 'cla':train_labels_cla}, \
        valid_images, {'reg':valid_labels_reg, 'cla':valid_labels_cla}, \
        test_images, {'reg':test_labels_reg, 'cla':test_labels_cla}


class cele_dataset(Dataset):
    def __init__(self, part, y, transform):
        self.part = part
        self.y = y
        self.transform = transform
        self.image = []
        for imgname in self.part:
            image = Image.open("../data/celeba/image/" + imgname)
            self.image.append(self.transform(image))
        self.image = torch.stack(self.image)
    def __getitem__(self, index):
        image = self.image[index]
        label = self.y[index]
        return image, label
    def __len__(self):
        return len(self.part)

class qmul_dataset(Dataset):
    def __init__(self, image, target=None):
        self.image = image
        self.y = target
    def __getitem__(self, index):
        return self.image[index], self.y[index]
    def __len__(self):
        return len(self.y)

class syn_dataset(Dataset):
    def __init__(self, x, y=None):
        self.image = x
        if self.image.dim == 1:
            self.image = self.image.reshape(-1, 1)
        self.y = y
    def __getitem__(self, index):
        if self.y == None:
            return self.image[index]
        else:
            return self.image[index], self.y[index]
    def __len__(self):
        return self.image.shape[0]


def sample_celeba(data, reg_name, cla_name, sample_size, scale=True, downsampling=True, alpha = 0.5):
    reg_data = []
    cla_data = []
    for task_name in reg_name:
        if scale: 
            data[[task_name]] = (data[[task_name]] - data[[task_name]].mean())/data[[task_name]].std()
        tmp_df = data[['name', task_name]].sample(sample_size, replace=False)
        tmp_df = tmp_df.reset_index(drop=True)
        reg_data.append({"name":tmp_df['name'].tolist(), task_name:tmp_df[task_name].tolist()})
    for task_name in cla_name:
        half_size = sample_size//2
        another_half = sample_size - half_size
        if downsampling:
            p = np.array([len(data[data[task_name] == 1]), len(data[data[task_name] == -1])])
            p = (p/len(data))**alpha
            q = p/np.sum(p)
            q[0] = int(q[0]*sample_size)
            q[1] = sample_size - q[0]
            tmp_df = pd.concat([data[data[task_name] == 1].sample(int(q[0]), replace=False), \
                data[data[task_name] == -1].sample(int(q[1]), replace=False)]).sample(frac=1, replace=False)[['name', task_name]].reset_index(drop=True)
        else:
            tmp_df = pd.concat([data[data[task_name] == 1].sample(half_size, replace=False), \
                data[data[task_name] == -1].sample(another_half, replace=False)]).sample(frac=1, replace=False)[['name', task_name]].reset_index(drop=True)

        cla_data.append({"name":tmp_df['name'].tolist(), task_name:tmp_df[task_name].tolist()})
    return reg_data, cla_data

def sample_dogcat(image, label, sample_size=None):
    reg_data = []
    cla_data = []
    for task_name, task_target in label.items():
        if sample_size == None:
            sample_size = len(task_target)
        sample_idx = np.random.choice(np.arange(len(task_target)), sample_size, replace=False)
        img = image[sample_idx]
        target = task_target[sample_idx]
        if task_name == 'reg':
            reg_data.append({'name':img, task_name:target})
        elif task_name == 'cla':
            cla_data.append({'name':img, task_name:target})
        else: raise ValueError("reg&cla names should be 'reg'/'cla'.")
    return reg_data, cla_data


def generate_clients(train_reg_data, train_cla_data, valid_reg_data, valid_cla_data, test_reg_data, test_cla_data, args):
    train_num_per_client = [len(train_reg_data[i]['name'])//args.num_client for i in range(args.num_reg_task)] + \
        [len(train_cla_data[i]['name'])//args.num_client for i in range(args.num_cla_task)]
    valid_num_per_client = [len(valid_reg_data[i]['name'])//args.num_client for i in range(args.num_reg_task)] + \
        [len(valid_cla_data[i]['name'])//args.num_client for i in range(args.num_cla_task)]
    test_num_per_client = [len(test_reg_data[i]['name'])//args.num_client for i in range(args.num_reg_task)] + \
        [len(test_cla_data[i]['name'])//args.num_client for i in range(args.num_cla_task)]
    clients = []
    for i in range(args.num_client):
        client = {"train_reg":[], "train_cla":[], "valid_reg":[], "valid_cla":[], "test_reg":[], "test_cla":[]}
        for j in range(args.num_reg_task):
            trainsize = train_num_per_client[j]
            train_picname = train_reg_data[j]['name'][i*trainsize:(i+1)*trainsize]
            train_piclabel = torch.tensor(train_reg_data[j][args.reg_task_name[j]][i*trainsize:(i+1)*trainsize], dtype=torch.float32)
            # celeba
            if args.dataset == 'celeba':
                client["train_reg"].append(DataLoader(cele_dataset(train_picname, train_piclabel, transform = args.transform), batch_size=args.batch_size, shuffle=True, drop_last=False))
            # dogcat
            elif args.dataset == 'dogcat':
                client["train_reg"].append(DataLoader(qmul_dataset(train_picname, train_piclabel), batch_size=args.batch_size, shuffle=True, drop_last=False))

            validsize = valid_num_per_client[j]
            valid_picname = valid_reg_data[j]['name'][i*validsize:(i+1)*validsize]
            valid_piclabel = torch.tensor(valid_reg_data[j][args.reg_task_name[j]][i*validsize:(i+1)*validsize], dtype=torch.float32)
            if args.dataset == 'celeba':
                client["valid_reg"].append(DataLoader(cele_dataset(valid_picname, valid_piclabel, transform = args.transform), batch_size=args.batch_size, shuffle=False, drop_last=False))
            elif args.dataset == 'dogcat':
                client["valid_reg"].append(DataLoader(qmul_dataset(valid_picname, valid_piclabel), batch_size=args.batch_size, shuffle=False, drop_last=False))
     
            testsize = test_num_per_client[j]
            test_picname = test_reg_data[j]['name'][i*testsize: (i+1)*testsize]
            test_piclabel = torch.tensor(test_reg_data[j][args.reg_task_name[j]][i*testsize:(i+1)*testsize], dtype=torch.float32)
            if args.dataset == 'celeba':
                client["test_reg"].append(DataLoader(cele_dataset(test_picname, test_piclabel, transform = args.transform), batch_size=args.batch_size, shuffle=False, drop_last=False))
            elif args.dataset == 'dogcat':
                client["test_reg"].append(DataLoader(qmul_dataset(test_picname, test_piclabel), batch_size=args.batch_size, shuffle=False, drop_last=False))


        for j in range(args.num_cla_task):
            trainsize = train_num_per_client[args.num_reg_task + j]
            train_picname = train_cla_data[j]['name'][i*trainsize: (i+1)*trainsize]
            train_piclabel = torch.tensor(train_cla_data[j][args.cla_task_name[j]][i*trainsize:(i+1)*trainsize]).long()
            # celeba
            if args.dataset == 'celeba':
                client["train_cla"].append(DataLoader(cele_dataset(train_picname, train_piclabel, transform = args.transform), batch_size=args.batch_size, shuffle=True, drop_last=False))
            elif args.dataset == 'dogcat':
                client["train_cla"].append(DataLoader(qmul_dataset(train_picname, train_piclabel), batch_size=args.batch_size, shuffle=True, drop_last=False))

            validsize = valid_num_per_client[args.num_reg_task + j]
            valid_picname = valid_cla_data[j]['name'][i*validsize: (i+1)*validsize]
            valid_piclabel = torch.tensor(valid_cla_data[j][args.cla_task_name[j]][i*validsize: (i+1)*validsize]).long()
            if args.dataset == 'celeba':
                client["valid_cla"].append(DataLoader(cele_dataset(valid_picname, valid_piclabel, transform = args.transform), batch_size=args.batch_size, shuffle=False, drop_last=False))
            elif args.dataset == 'dogcat':
                client["valid_cla"].append(DataLoader(qmul_dataset(valid_picname, valid_piclabel), batch_size=args.batch_size, shuffle=False, drop_last=False))

            testsize = test_num_per_client[args.num_reg_task + j]
            test_picname = test_cla_data[j]['name'][i*testsize: (i+1)*testsize]
            test_piclabel = torch.tensor(test_cla_data[j][args.cla_task_name[j]][i*testsize: (i+1)*testsize]).long()
            if args.dataset == 'celeba':
                client["test_cla"].append(DataLoader(cele_dataset(test_picname, test_piclabel, transform = args.transform), batch_size=args.batch_size, shuffle=False, drop_last=False))
            elif args.dataset == 'dogcat':
                client["test_cla"].append(DataLoader(qmul_dataset(test_picname, test_piclabel), batch_size=args.batch_size, shuffle=False, drop_last=False))

        clients.append(client)   
    return clients 