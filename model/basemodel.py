import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b2,EfficientNet_B2_Weights, densenet169, DenseNet169_Weights, shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights
from torchvision.models import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights


## backbone ##
# we use pretrained resnet-18 as backbone and report results in main main experiments. 
# Besides, we provide abalation study about different options about backbone (efficientnet, densenet, shufflenet, regnet).
class CNNModel(nn.Module):
    def __init__(self, num_dim, freeze="all", mlp=True):
        r'''
        num_dim: output dimension of backbone as feature extractor
        freeze: options in ['all', 'bottom', 'none'], freeze some component of backbone during training
        mlp: bool type, whether or not to add linear layer after backbone
        '''
        super(CNNModel, self).__init__()
        self.model = resnet18(weights = ResNet18_Weights.DEFAULT)
        if mlp:
            self.model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=num_dim))
        else:
            self.model.fc = nn.Sequential()
        self.freeze = freeze
        self.num_dim = num_dim
        if freeze == "all":
            for param in self.model.parameters():
                param.requires_grad = False
        elif freeze == 'bottom':
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.layer4.parameters():
                param.requires_grad = True
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif freeze == 'none':
            pass
        else:
            raise ValueError("freeze only support all, bottom, none")
    def forward(self, x):
        return self.model(x)

    # def __init__(self, num_dim, freeze="all", mlp=True):
    #     super(CNNModel, self).__init__()
    #     self.model = efficientnet_b2(weights = EfficientNet_B2_Weights.DEFAULT)
    #     if mlp:
    #         self.model.classifier = nn.Sequential(nn.Linear(in_features=1408, out_features=num_dim))
    #     else:
    #         self.model.classifier = nn.Sequential()
    #     self.freeze = freeze
    #     self.num_dim = num_dim
    #     if freeze == "all":
    #         for param in self.model.parameters():
    #             param.requires_grad = False
    #     elif freeze == 'bottom':
    #         for param in self.model.parameters():
    #             param.requires_grad = False
    #         for param in self.model.features[8].parameters():
    #             param.requires_grad = True
    #         for param in self.model.classifier.parameters():
    #             param.requires_grad = True
    #     elif freeze == 'none':
    #         pass
    #     else:
    #         raise ValueError("freeze only support all, bottom, none")
    # def forward(self, x):
    #     return self.model(x)
    
    # def __init__(self, num_dim, freeze="all", mlp=True):
    #     super(CNNModel, self).__init__()
    #     self.model = densenet169(weights = DenseNet169_Weights.DEFAULT)
    #     if mlp:
    #         self.model.classifier = nn.Sequential(nn.Linear(in_features=1664, out_features=num_dim))
    #     else:
    #         self.model.classifier = nn.Sequential()
    #     self.freeze = freeze
    #     self.num_dim = num_dim
    #     if freeze == "all":
    #         for param in self.model.parameters():
    #             param.requires_grad = False
    #     elif freeze == 'none':
    #         pass
    #     else:
    #         raise ValueError("freeze only support all, bottom, none")
    # def forward(self, x):
    #     return self.model(x)
    
    # def __init__(self, num_dim, freeze="all", mlp=True):
    #     super(CNNModel, self).__init__()
    #     self.model = shufflenet_v2_x2_0(weights = ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
    #     if mlp:
    #         self.model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=num_dim))
    #     else:
    #         self.model.fc = nn.Sequential()
    #     self.freeze = freeze
    #     self.num_dim = num_dim
    #     if freeze == "all":
    #         for param in self.model.parameters():
    #             param.requires_grad = False
    #     elif freeze == 'none':
    #         pass
    #     else:
    #         raise ValueError("freeze only support all, bottom, none")
    # def forward(self, x):
    #     return self.model(x)
    
    # def __init__(self, num_dim, freeze="all", mlp=True):
    #     super(CNNModel, self).__init__()
    #     self.model = regnet_y_1_6gf(weights = RegNet_Y_1_6GF_Weights.IMAGENET1K_V1)
    #     if mlp:
    #         self.model.fc = nn.Sequential(nn.Linear(in_features=888, out_features=num_dim))
    #     else:
    #         self.model.fc = nn.Sequential()
    #     self.freeze = freeze
    #     self.num_dim = num_dim
    #     if freeze == "all":
    #         for param in self.model.parameters():
    #             param.requires_grad = False
    #     elif freeze == 'none':
    #         pass
    #     else:
    #         raise ValueError("freeze only support all, bottom, none")
    # def forward(self, x):
    #     return self.model(x)

class synmodel1d(nn.Module):
    def __init__(self) -> None:
        super(synmodel1d, self).__init__()
        pass
    def forward(self, x):
        return x
