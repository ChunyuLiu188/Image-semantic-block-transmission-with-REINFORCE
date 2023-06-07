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

EPOCHS = 20
STOP_STEP = 5
N_CLASSES = 45
LR = 1e-4
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 64
CLS_SAVE_PATH = "pretrained/classificer.pth"


seed = 2023
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

logging.basicConfig(format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='log/train_cls.log',)
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger()



def train(model, optimizer, loss_fn):
    train_dataset = BaseDataset("data/train.pkl")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = BaseDataset("data/val.pkl")
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    best_acc = 0
    stop_step = 0
    for epoch in range(EPOCHS):
        #*************** train **************
        model.train()
        epoch_loss = []
        for pictures, labels in tqdm(train_dataloader):
            pictures, labels = pictures.to(device), labels.to(device)
            predict = model(pictures)
            loss = loss_fn(predict, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        epoch_loss = np.mean(epoch_loss)
        #************ test *****************
        epoch_acc = test(model, val_dataloader, pretrained=False)
        
        # #***************** early stopping *************
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            stop_step = 0
            torch.save(model.state_dict(), CLS_SAVE_PATH)
        else: stop_step += 1
        if stop_step == STOP_STEP:
            break
        
        logger.info(f"Epoch: {epoch}, loss: {epoch_loss}, val_acc: {epoch_acc}")
def test(model, dataloader, pretrained=False):  
    if pretrained == True:
        model.load_state_dict(torch.load(CLS_SAVE_PATH))     
    model.eval()
    epoch_acc = []
    for pictures, labels in dataloader:
        pictures, labels = pictures.to(device), labels.to(device)
        predict = model(pictures).argmax(dim=-1)
        acc = accuracy_score(predict.cpu(), labels.cpu())
        epoch_acc.append(acc.item())
    epoch_acc = np.mean(epoch_acc)
    return epoch_acc
         
if __name__ == "__main__":
    model = Classification(n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    train(model, optimizer, loss_fn)
    
    #*************** test **************************************
    # test_dataset = BaseDataset("data/val.pkl")
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # acc = test(model, test_dataloader, pretrained=True)
    # print(acc) # 0.9313