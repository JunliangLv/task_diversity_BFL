## Task Diversity in Bayesian Federated Learning: Simultaneous Processing of Classification and Regression

This repository contains the official implementation of Task Diversity in Bayesian Federated Learning: Simultaneous Processing of Classification and Regression (denoted as pFed-St and pFed-Mul) in our paper. 

### Abstract

This work addresses a key limitation in current federated learning approaches, which predominantly focus on homogeneous tasks, neglecting the task diversity on local devices. We propose a principled integration of multi-task learning using multi-output Gaussian processes (MOGP) at the local level and federated learning at the global level. MOGP handles correlated classification and regression tasks, offering a Bayesian non-parametric approach that naturally quantifies uncertainty. The central server aggregates the posteriors from local devices, updating a global MOGP prior redistributed for training local models until convergence. Challenges in performing posterior inference on local devices are addressed through the $P\acute{o}lya-Gamma$ augmentation technique and mean-field variational inference, enhancing computational efficiency and convergence rate. Experimental results on both synthetic and real data demonstrate superior predictive performance, OOD detection, uncertainty calibration and convergence rate, highlighting the method's potential in diverse applications. 

### Setup
##### Installation

```
pip install -r requirements.txt
```
##### Datasets
We consider one synthetic dataset, with hyperparameters reported in our paper. The related code is provided in ./data/synthetic/generate_synthetic_data.ipynb.


Also, we consider two real-world datasets, namely [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Dogcat](https://www.kaggle.com/datasets/tongpython/cat-and-dog). The complete data can be downloaded from corresponding websites and placed in folder "./data". We recommend that data is reorganized as

- **CelebA**: 
``` 
./data/celeba/
├── Anno/
└── image/
```
- **Dogcat**:
```
./data/dogcat/
├── train/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```
##### Resources
We did all experiments in this paper using servers with two GPUs (NVIDIA TITAN V with 12GB memory), two CPUs (each with 8 cores, Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz), and 251GB memory.

### Experiments

All commands for running the paper experiments (pFed-St and pFed-Mul with synthetic, CelebA and Dogcat datasets) can be found in ./data/paper_experiment. 
```
# For synthetic dataset
cd ./paper_experiment/
sh synthetic.sh

# For celeba dataset
cd ./paper_experiment/
sh celeba.sh

# For dogcat dataset
cd ./paper_experiment/
sh dogcat.sh
```

