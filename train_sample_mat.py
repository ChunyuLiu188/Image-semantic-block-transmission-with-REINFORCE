import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from Models import Classification, BaseDataset
from torch.utils.data import DataLoader
import numpy as np
import logging
import pickle as pk
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from Models import sample
EPOCHS = 100
STOP_STEP = 5
N_CLASSES = 45
LR = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 64
SAMPLE_SAVE_PATH = "pretrained/sample.pth"


seed = 2023
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

logging.basicConfig(format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='log/train_sample_mat.log',)
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger()

# class encoder(nn.Module):
#     def __init__(self):
#         super(encoder, self).__init__()
#         self.encoder = nn.Sequential(
#                                      nn.Conv2d(3, 64, kernel_size=(3, 3), stride=2, bias=False),
#                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                                      nn.ReLU(),
#                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2, bias=False, padding=1),
#                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                                      nn.ReLU(),
#                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, bias=False),
#                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                                      nn.ReLU(),
#                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, bias=False),
#                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                                      nn.ReLU(),
#                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, bias=False),
#                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                                      nn.ReLU(),
#                                      nn.Conv2d(64, 3, kernel_size=(3, 3), bias=False))
        
#     def forward(self, x):
#         out = self.encoder(x)
#         return out

# class decoder(nn.Module):
#     def __init__(self):
#         super(decoder, self).__init__()
#         self.decoder = nn.Sequential(nn.ConvTranspose2d(3, 64, kernel_size=(3, 3), stride=1, bias=False),
#                                      nn.ReLU(),
#                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                                      nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=1, bias=False),
#                                      nn.ReLU(),
#                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                                      nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=1, bias=False),
#                                      nn.ReLU(),
#                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                                      nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=1, bias=False),
#                                      nn.ReLU(),
#                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                                      nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=2, bias=False),
#                                      nn.ReLU(),
#                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                                      nn.ConvTranspose2d(64, 3, kernel_size=(3, 3), stride=2, bias=False),
#                                     )
#         # for i in range(18):
#         #     self.decoder.add_module(f"{i}_c", nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2, bias=False))
#         #     self.decoder.add_module(f"{i}+b", nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
#         #     self.decoder.add_module(f"{i}_r", nn.ReLU())
#         # self.decoder.add_module("last", nn.Conv2d(64, 3, kernel_size=(3, 3), bias=False))
        
#     def forward(self, x):
#         out = self.decoder(x)
#         return out


        
        
        
        
train_dataset = BaseDataset("data/train.pkl")
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = BaseDataset("data/val.pkl")
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
best_acc = 0
stop_step = 0
model1 = sample().to(device)
model1.load_state_dict(torch.load(SAMPLE_SAVE_PATH))
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model1.parameters(), LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
best_acc = 1e4
stop_step = 0

for epoch in range(EPOCHS):
        #*************** train **************
    model1.eval()
  
    
    epoch_loss = []
    for pictures, _ in tqdm(val_dataloader):
        pictures = pictures.to(device)
        predict = model1(pictures)
        
        
        loss = loss_fn(predict, pictures)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    epoch_loss = np.mean(epoch_loss)
    print(epoch_loss)
    
    if epoch % 1 == 0:
        p = predict[0].permute(1, 2, 0).cpu().detach().numpy()
        p = 0.5 * (p + 1)
        p = np.clip(p, 0, 1)
        plt.imsave("rec.png", p)
    if epoch_loss <= best_acc:
        best_acc = epoch_loss
        stop_step = 0
        torch.save(model1.state_dict(), SAMPLE_SAVE_PATH)
    else: stop_step += 1
    if stop_step == STOP_STEP:
        break
    scheduler.step()