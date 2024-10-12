from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
dogcat_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
celeba_transform = transforms.Compose([
            transforms.Resize([256]),
            transforms.CenterCrop([224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
un_transform = transforms.Compose([
               transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
               transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
               transforms.ToPILImage()
        ]) 
class ood_dataset(Dataset):
    def __init__(self, x):
        super().__init__()
        self.x = x
    def __getitem__(self, index):
        return self.x[index]
    def __len__(self):
        return len(self.x)

def mask_pic(path):
    img = Image.open(path)
    width, height = img.size
    x_start = width * 0.05
    x_end = width * 0.95
    y_start = height * 0.05
    y_end = height * 0.95

    draw = ImageDraw.Draw(img)
    draw.rectangle([x_start, y_start, x_end, y_end], fill='black')
    return img

def get_ood_sample(dataset, sample_size):
    if dataset == 'dogcat':
        dog = []
        cat = []
        for i in range(sample_size//2):
            img_cat = Image.open(os.path.join("../data/dogcat/train/cats", f"cat.{i+1}.jpg"))
            img_dog = Image.open(os.path.join("../data/dogcat/train/dogs", f"dog.{i+1}.jpg"))
            cat.append(dogcat_transform(img_cat))
            dog.append(dogcat_transform(img_dog))
        ood1 = dogcat_transform(mask_pic("../data/dogcat/test/cats/cat.5000.jpg"))
        ood2 = dogcat_transform(mask_pic("../data/dogcat/test/dogs/dog.5000.jpg"))
        dog.append(ood1)
        cat.append(ood2)

        dog, cat = torch.stack(dog), torch.stack(cat)
        ood_sample = torch.concat([dog[np.random.choice(np.arange(len(dog)), size=len(dog), replace=False)], \
                                   cat[np.random.choice(np.arange(len(cat)), size=len(cat), replace=False)]])
    
    if dataset == 'celeba':
        part = pd.read_csv("../data/celeba/Anno/list_eval_partition.txt", sep='\\s+', header=None)
        part.columns = ['image name', 'partition']
        sample_name = part[part['partition'] == 0].sample(sample_size)['image name'].tolist()
        ood_sample = []
        for sample in sample_name:
            img = Image.open(os.path.join("../data/celeba/image/", sample))
            ood_sample.append(celeba_transform(img))
        ood_name = part[part['partition'] == 2].sample(2)['image name'].tolist()
        for ood in ood_name:
            ood_sample.append(celeba_transform(mask_pic(os.path.join("../data/celeba/image", ood)))) 
        ood_sample = torch.stack(ood_sample)
        ood_sample = ood_sample[np.random.choice(np.arange(len(ood_sample)), size=len(ood_sample), replace=False)]
    return ood_dataset(ood_sample)

def get_ood_density(dataset, num_sample):
    train_images = []
    test_images = []
    ood_images = []
    if dataset == 'dogcat':
        for i in range(num_sample//2):
            img = Image.open(os.path.join("../data/dogcat/train/dogs", f"dog.{i+1}.jpg"))
            train_images.append(dogcat_transform(img))
            img = Image.open(os.path.join("../data/dogcat/train/cats", f"cat.{i+1}.jpg"))
            train_images.append(dogcat_transform(img))

            img = Image.open(os.path.join("../data/dogcat/test/dogs/", f"dog.{4000+i+1}.jpg"))
            test_images.append(dogcat_transform(img))
            img = Image.open(os.path.join("../data/dogcat/test/cats/", f"cat.{4000+i+1}.jpg"))
            test_images.append(dogcat_transform(img))

            img = Image.open(os.path.join("../data/dogcat/test/dogs/", f"dog.{4000+i+1}.jpg"))
            img = img.filter(ImageFilter.GaussianBlur(20))
            ood_images.append(dogcat_transform(img))
            img = Image.open(os.path.join("../data/dogcat/test/cats/", f"cat.{4000+i+1}.jpg"))
            img = img.filter(ImageFilter.GaussianBlur(20))
            ood_images.append(dogcat_transform(img))

    if dataset == 'celeba':
        part = pd.read_csv("../data/celeba/Anno/list_eval_partition.txt", sep='\\s+', header=None)
        part.columns = ['image name', 'partition']
        train_list = part[part['partition'] == 0].sample(num_sample)['image name'].tolist()
        test_list = part[part['partition'] == 2].sample(num_sample)['image name'].tolist()
        for train_imgname in train_list:
            img = Image.open(os.path.join("../data/celeba/image/", train_imgname))
            train_images.append(celeba_transform(img))
        for test_imgname in test_list:
            img = Image.open(os.path.join("../data/celeba/image/", test_imgname))
            test_images.append(celeba_transform(img))
            img = img.filter(ImageFilter.GaussianBlur(20))
            ood_images.append(celeba_transform(img))
    train_images = torch.stack(train_images)
    test_images = torch.stack(test_images)
    ood_images = torch.stack(ood_images)
    return ood_dataset(train_images), ood_dataset(test_images), ood_dataset(ood_images)