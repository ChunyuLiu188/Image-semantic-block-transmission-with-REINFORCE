from torchvision.models import resnet152, resnet18
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle as pk
from PIL import Image
import torch.nn.functional as F

SAMPLE_SAVE_PATH = "pretrained/sample.pth"

class Policy(nn.Module):
    def __init__(self, n_actions):
        super(Policy, self).__init__()
        model = resnet18(pretrained=True)
        self.f_features = nn.Sequential(*list(m for m in model.children())[:-1])
          
        self.f_gain = nn.Sequential(nn.Linear(1, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 512, bias=False)
                                    )
        self.f_decision = nn.Sequential(nn.Linear(512, 128, bias=False),
                                        nn.Tanh(),
                                        nn.Linear(128, n_actions, bias=False)   
                                    ) 
    def forward(self, inputs, gains):
        out = self.f_features(inputs)
        out = torch.flatten(out, 1)
        gain_feature = self.f_gain(gains)
        out = self.f_decision(out + gain_feature)
        return out

class Classification(nn.Module):
    def __init__(self, n_classes):
        super(Classification, self).__init__()
        self.model = resnet152(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, n_classes)
        
    def forward(self, inputs):
        out = self.model(inputs)
        return out  
    
transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])  

class BaseDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            data = pk.load(f)
            self.pictures = data[0]
            self.labels = data[1]
                
    def __getitem__(self, index):
        
        picture = self.pictures[index]
        label = self.labels[index]
        return picture, label
    
    def __len__(self):
        return len(self.labels)

class sample(nn.Module):
    def __init__(self):
        super(sample, self).__init__()
        self.k = nn.Conv2d(in_channels=3, out_channels=231, kernel_size=16, stride=16, bias=False)
        self.k_auxiliary = nn.ConvTranspose2d(231, 3, kernel_size=16,  stride=16, bias=False) 
        
    def forward(self, x):
        out = self.k(x)
        out = self.k_auxiliary(out)
        return out

class rec(nn.Module):
    def __init__(self):
        super(rec, self).__init__()
    
        self.model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, bias=False, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 3, kernel_size=(3, 3), stride=1, bias=False, padding=1),
                                )
    def forward(self, x):
        out = self.model(x)
        out = x - out
        
        
        # e = self.sample_model.k_auxiliary(y - self.sample_model.k(x))  
        # z = e + self.residual2(z)
        # x = self.denoising2(x + z)
        
        # e = self.sample_model.k_auxiliary(y - self.sample_model.k(x))  
        # z = e + self.residual3(z)
        # x = self.denoising3(x + z)
        return out

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out




