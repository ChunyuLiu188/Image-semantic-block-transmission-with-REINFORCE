import torch
from Models import Classification, BaseDataset, Policy
from torch.utils.data import DataLoader
import numpy as np
import logging
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from Models import sample, rec, DnCNN

LR_REC = 1e-4
EPOCHS = 100
N_BLOCKS = 16
ALPHA = 0.7
N_CLASSES = 45
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
GAIN_LIST = np.array([-30, -20, -10, 0, 10, 20, 30])
R_DB = np.array([105, 355, 675, 1006, 1338, 1670, 2003])
CLS_SAVE_PATH = "pretrained/classificer.pth"
POLICY_SAVE_PATH = "pretrained/policy.pth"
SAMPLE_SAVE_PATH = "pretrained/sample.pth"
REC_SAVE_PATH = "pretrained/rec.pth"


seed = 2023
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

logging.basicConfig(format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename="log/train_policy.log")
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger()




def train(policy_model, classification_model, sample_model, rec_model):
    #***************reading data *******************
    train_dataset = BaseDataset("data/train.pkl")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataset = BaseDataset("data/test.pkl")
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer_rec = torch.optim.AdamW(rec_model.parameters(), LR_REC)
    optimizer_cls = torch.optim.AdamW(filter(lambda x : x.requires_grad, classification_model.parameters()), LR_REC)
    for epoch in range(EPOCHS):
    # #********************** train deep cs **********************************************************************
        rec_model.train()
        
        
        for pictures, _ in tqdm(train_dataloader):
            
            
            pictures = pictures.to(device)
            y_send = sample_model.k(pictures) #[batch, c, 16, 16]
            noise = torch.tensor(np.random.normal(0, 0.01,size=(BATCH_SIZE,231,16,16)), requires_grad=False, dtype=torch.float32).to(device)
            y_receive = noise + y_send
            
            x = sample_model.k_auxiliary(y_receive) # initial rec x0 [batch, 3, 256, 256]
            
            rec_pictures = rec_model(x)
            
            loss_rec = nn.MSELoss()(rec_pictures, pictures) / 2*BATCH_SIZE
            optimizer_rec.zero_grad()
            loss_rec.backward()
            optimizer_rec.step
        
            
        torch.save(rec_model.state_dict(), REC_SAVE_PATH)
        # #*******************  fineturn cls ***************************************************************
        classification_model.train()
        for pictures, labels in tqdm(train_dataloader):
            pictures, labels = pictures.to(device), labels.to(device)
            y_send = sample_model.k(pictures) #[batch, c, 16, 16]
            noise = torch.tensor(np.random.normal(0, 0.01,size=(BATCH_SIZE,231,16,16)), requires_grad=False, dtype=torch.float32).to(device)
            y_receive = noise + y_send
            
            x = sample_model.k_auxiliary(y_receive) # initial rec x0 [batch, 3, 256, 256]
            rec_model.eval()
            with torch.no_grad():
                rec_pictures = rec_model(x)
            predict = classification_model(rec_pictures)
            loss_cls = nn.CrossEntropyLoss()(predict, labels)
            optimizer_cls.zero_grad()
            loss_cls.backward()
            optimizer_cls.step()
            
        torch.save(classification_model.state_dict(), CLS_SAVE_PATH)         
    #************************************* train policy **********************************************************************
        policy_model.train()
        
        for stage in range(3):
            logger.info(f"Stage {stage} Training")
            if stage == 0: # stage 1 : freeze the paremeters in f_gain 
                for para in policy_model.f_gain.parameters():
                    para.requires_grad = False
            elif stage == 1:# stage 2 : freeze the paremeters in resnet
                for para in policy_model.f_features.parameters():
                    para.requires_grad = False
                for para in policy_model.f_gain.parameters():
                    para.requires_grad = True
            elif stage == 2:
                for para in policy_model.f_features.parameters():
                    para.requires_grad = True
            optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, policy_model.parameters()), lr=LR)
            
            for eposide in range(1):
                Reward = []
                Loss = []
                Acc = []
                T_Block = []
               
                for pictures, labels in tqdm(train_dataloader):
                    pictures, labels = pictures.to(device), labels.to(device)
                    gain_indices = np.random.choice(range(len(GAIN_LIST)), size=BATCH_SIZE, replace=True)
                    gains = torch.tensor(((GAIN_LIST[gain_indices] + 30) / 60).reshape(-1, 1), dtype=torch.float32, requires_grad=False).to(device)
                    
                    probs = torch.sigmoid(policy_model(pictures, gains))
                    # during the early training stage, more exploration
                    alpha_hp = np.clip(ALPHA + 0.0026*epoch, 0.7, 0.95)
                    probs = probs * alpha_hp + (1-alpha_hp) * (1-probs)
                    # take actions
                    distr = Bernoulli(probs)
                    
                    policy_sample = distr.sample()
                    #[BATCH, N_ACTIONS] -> [BATCH, sqrt(N_BLOCKS), sqrt(N_BLOCKS)]
                    policy_sample2 = policy_sample.view(BATCH_SIZE, int(np.sqrt(N_BLOCKS)), int(np.sqrt(N_BLOCKS)))
                    #[BATCH, sqrt(N_BLOCKS), sqrt(N_BLOCKS)] -> [BATCH, H, W]
                    policy_sample2 = policy_sample2.repeat_interleave(64, 1).repeat_interleave(64, 2)
                    # [BATCH, H, W] -> [BATCH, 3, H, W]
                    policy_sample2 = policy_sample2.unsqueeze(1).repeat(1, 3, 1, 1)
                    # the semantic blocks
                    pictures_bolcks = policy_sample2 * pictures 
                    pictures_bolcks_sample = sample_model.k(pictures_bolcks)
                    p1 = pictures_bolcks[0].permute(1,2,0).detach().cpu().numpy()
                    p1 = 0.5 * (p1 + 1)
                    p1 = np.clip(p1, 0, 1)
                    
                    # add noise
                    noise = torch.tensor(np.random.normal(0, 0.01,size=(BATCH_SIZE,231,16,16)), requires_grad=False, dtype=torch.float32).to(device)
                    pictures_bolcks_sample += noise
                    pictures_bolcks_receive = sample_model.k_auxiliary(pictures_bolcks_sample)
                    rec_model.eval()
                    with torch.no_grad():
                        pictures_bolcks = rec_model(pictures_bolcks_receive)
                    p2 = pictures_bolcks[0].permute(1,2,0).cpu().detach().numpy()
                    p2 = 0.5 * (p2 + 1)
                    p2 = np.clip(p2, 0, 1)
                    p1 = np.concatenate((p1, p2), 1)
                    plt.imsave("rec.png", p1)
       
                    """
                    get reward
                    """
                    reward, acc, n_b = get_reward(classification_model, pictures_bolcks, labels, policy_sample, gain_indices)
                    Acc.append(acc)
                    Reward.append(reward.item())
                    T_Block.append(n_b.item())
                    """Rm,n is the reward expectation when each sub-action is performed as expectations of the output policy p. This
                    action directly selects the image blocks according to the larger value of the output policy
                    """
                    # policy_map = probs.data.clone()
                    # policy_map[policy_map<0.5] = 0.0
                    # policy_map[policy_map>=0.5] = 1.0
                    
                    
                    # policy_sample_e2 = policy_map.view(BATCH_SIZE, int(np.sqrt(N_BLOCKS)), int(np.sqrt(N_BLOCKS)))
                    # #[BATCH, sqrt(N_BLOCKS), sqrt(N_BLOCKS)] -> [BATCH, H, W]
                    # policy_sample_e2 = policy_sample_e2.repeat_interleave(64, 1).repeat_interleave(64, 2)
                    # # [BATCH, H, W] -> [BATCH, 3, H, W]
                    # policy_sample_e2 = policy_sample_e2.unsqueeze(1).repeat(1, 3, 1, 1)
                    # # the semantic blocks
                    # pictures_bolcks = policy_sample2 * pictures 
                    # reward_e, _ , _= get_reward(classification_model, pictures_bolcks, labels, policy_map, gain_indices)
                
                    # advantage = reward - reward_e

                    # Find the loss for only the policy network
                    loss = -distr.log_prob(policy_sample)
                    loss = loss * reward.expand_as(policy_sample)
                    loss = loss.mean()
                    Loss.append(loss.item())
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                V_Reward = []
                V_Acc = []
                V_Blocks = []
                for pictures, labels in tqdm(val_dataloader):
                    pictures, labels = pictures.to(device), labels.to(device)
                    gain_indices = np.random.choice(range(len(GAIN_LIST)), size=BATCH_SIZE, replace=True)
                    gains = torch.tensor(((GAIN_LIST[gain_indices] + 30) / 60).reshape(-1, 1), dtype=torch.float32, requires_grad=False).to(device)
                    
                    with torch.no_grad():
                        probs = torch.sigmoid(policy_model(pictures, gains))
                    alpha_hp = np.clip(ALPHA + 0.0026*epoch, 0.7, 0.95)
                    probs = probs * alpha_hp + (1-alpha_hp) * (1-probs)
                    distr = Bernoulli(probs)
                            
                    policy_sample = distr.sample()
                    #[BATCH, N_ACTIONS] -> [BATCH, sqrt(N_BLOCKS), sqrt(N_BLOCKS)]
                    policy_sample2 = policy_sample.view(BATCH_SIZE, int(np.sqrt(N_BLOCKS)), int(np.sqrt(N_BLOCKS)))
                    #[BATCH, sqrt(N_BLOCKS), sqrt(N_BLOCKS)] -> [BATCH, H, W]
                    policy_sample2 = policy_sample2.repeat_interleave(64, 1).repeat_interleave(64, 2)
                    # [BATCH, H, W] -> [BATCH, 3, H, W]
                    policy_sample2 = policy_sample2.unsqueeze(1).repeat(1, 3, 1, 1)
                    # the semantic blocks
                    pictures_bolcks = policy_sample2 * pictures
                    pictures_bolcks_sample = sample_model.k(pictures_bolcks)
                    noise = torch.tensor(np.random.normal(0, 0.01,size=(BATCH_SIZE,231,16,16)), requires_grad=False, dtype=torch.float32).to(device)
                    pictures_bolcks_sample += noise
                    pictures_bolcks_receive = sample_model.k_auxiliary(pictures_bolcks_sample)
                    rec_model.eval()
                    with torch.no_grad():
                        pictures_bolcks = rec_model(pictures_bolcks_receive)
                    
                    """
                    get reward
                    """
                    reward, acc, n_block = get_reward(classification_model, pictures_bolcks, labels, policy_sample, gain_indices)
                    V_Acc.append(acc)
                    V_Reward.append(reward.item())
                    V_Blocks.append(n_block.item())
                  
                 
                logger.info(f"Epoch:{epoch}, Reward:{np.mean(Reward)}, Acc:{np.mean(Acc)}, N_blocks:{np.mean(T_Block)}, Loss:{np.mean(Loss)}, Val_Reward:{np.mean(V_Reward)}, Val_Acc:{np.mean(V_Acc)}, VAL_N_blocks:{np.mean(V_Blocks)}")
            torch.save(policy_model.state_dict(), POLICY_SAVE_PATH)
            
def get_reward(classification_model, pictures_bolcks, labels, policy_sample, gain_indices):
    with torch.no_grad():
        predictions = classification_model(pictures_bolcks).argmax(dim=-1)
    acc = accuracy_score(predictions.cpu(), labels.cpu())
    eq = (torch.eq(predictions, labels) * 1).view(-1, 1)           
    n_blocks = policy_sample.sum(dim=-1).view(-1, 1)
    r_dbs = torch.tensor(R_DB[gain_indices].reshape(-1, 1), dtype=torch.float32, requires_grad=False).to(device)
    latency = (n_blocks * 8 * 64 * 64 * 0.3 / r_dbs) * eq * 1e-3
    latency = (1 / (1 + 2*latency)) * eq
    latency = latency[latency>0]
    reward = torch.sum(latency) + len(eq[eq==0]) * 0.15
    
    return reward, acc, torch.mean(n_blocks)

# def test(policy_model, classification_model, sample_model, rec_model):
#     V_Reward = []
#     V_Acc = []
#     V_Blocks = []
#     test_dataset = BaseDataset("test/train.pkl")
#     test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#     for pictures, labels in tqdm(test_dataloader):
#         pictures, labels = pictures.to(device), labels.to(device)
#                     gain_indices = np.random.choice(range(len(GAIN_LIST)), size=BATCH_SIZE, replace=True)
#                     gains = torch.tensor(((GAIN_LIST[gain_indices] + 30) / 60).reshape(-1, 1), dtype=torch.float32, requires_grad=False).to(device)
                    
#                     with torch.no_grad():
#                         probs = torch.sigmoid(policy_model(pictures, gains))
#                     alpha_hp = np.clip(ALPHA + 0.0026*epoch, 0.7, 0.95)
#                     probs = probs * alpha_hp + (1-alpha_hp) * (1-probs)
#                     distr = Bernoulli(probs)
                            
#                     policy_sample = distr.sample()
#                     #[BATCH, N_ACTIONS] -> [BATCH, sqrt(N_BLOCKS), sqrt(N_BLOCKS)]
#                     policy_sample2 = policy_sample.view(BATCH_SIZE, int(np.sqrt(N_BLOCKS)), int(np.sqrt(N_BLOCKS)))
#                     #[BATCH, sqrt(N_BLOCKS), sqrt(N_BLOCKS)] -> [BATCH, H, W]
#                     policy_sample2 = policy_sample2.repeat_interleave(64, 1).repeat_interleave(64, 2)
#                     # [BATCH, H, W] -> [BATCH, 3, H, W]
#                     policy_sample2 = policy_sample2.unsqueeze(1).repeat(1, 3, 1, 1)
#                     # the semantic blocks
#                     pictures_bolcks = policy_sample2 * pictures
#                     pictures_bolcks_sample = sample_model.k(pictures_bolcks)
#                     noise = torch.tensor(np.random.normal(0, 0.01,size=(BATCH_SIZE,231,16,16)), requires_grad=False, dtype=torch.float32).to(device)
#                     pictures_bolcks_sample += noise
#                     pictures_bolcks_receive = sample_model.k_auxiliary(pictures_bolcks_sample)
#                     rec_model.eval()
#                     with torch.no_grad():
#                         pictures_bolcks = rec_model(pictures_bolcks_receive)
                    
#                     """
#                     get reward
#                     """
#                     reward, acc, n_block = get_reward(classification_model, pictures_bolcks, labels, policy_sample, gain_indices)
#                     V_Acc.append(acc)
#                     V_Reward.append(reward.item())
#                     V_Blocks.append(n_block.item())
                  

    
         
if __name__ == "__main__":
    #*****************classification model*************
    classification_model = Classification(n_classes=N_CLASSES).to(device)
    classification_model.load_state_dict(torch.load(CLS_SAVE_PATH))
    for m in list(classification_model.model.children())[:-3]:
        for p in m.parameters():
            p.requires_grad = False
    #*****************sample model*************   
    sample_model = sample().to(device)
    sample_model.load_state_dict(torch.load(SAMPLE_SAVE_PATH))
    sample_model.eval()
    for para in sample_model.parameters():
        para.requires_grads = False
    rec_model = rec().to(device)
    policy_model = Policy(n_actions=N_BLOCKS).to(device)
    
    train(policy_model, classification_model, sample_model, rec_model)