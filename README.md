# -----# 数据集

使用NWPU-RESISC45数据集，NWPU-RESISC45遥感数据集是由西北工业大学公布的用于遥感图像场景分类的大规模公开数据集，包含45类场景，各类别场景样本展示如下图所示。&#x20;

![](https://img-blog.csdnimg.cn/3d3413264eed450f82f1ed2391ba8afa.png#pic_center)

&#x20;  上图中具体每个类别为(1) airplane、(2) airport、(3) baseball diamond、(4) basketball court、 (5) beach 、 (6) bridge、 (7) chaparral、 (8) church、 (9) circular farmland、 (10) cloud、 (11) commercial area、(12) dense residential、(13) desert、(14) forest、(15) freeway、(16) golf course、(17) ground track field、(18) harbor、(19) industrial area、(20) intersection、(21) island、 (22) lake、(23) meadow 、(24) medium residential、(25) mobilehome park 、(26) mountain、 (27) overpass、(28) palace、(29) parking lot、(30) railway、(31) railway station、(32) rectangular farmland、(33) river 、(34) roundabout、(35) runway、(36) sea ice、(37) ship、(38) snowberg、 (39) sparse residential、(40) stadium、(41) storage tank、(42) tennis court、(42) terrace、(44) thermal power station、(45) wetland，其中每个类别各包含700 张遥感图像，整个数据集一共31500张图像，每张图像的大小为256 × 256，具有规模大、图像中所含信息丰富等特点。

## 数据预处理

按8：1：1划分训练集、验证集和测试集。

图片transform

```python
transformer = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(), #转换为tensor
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) #归一化到【-1，1】
])  
```

最后将所有的pictures和对应的labels用pkl文件保存。

读取数据dataset类为：

```python
class BaseDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f: # path为train.pkl或val.pkl或test.pkl的路径
            data = pk.load(f)
            self.pictures = data[0]
            self.labels = data[1]
                
    def __getitem__(self, index):
        
        picture = self.pictures[index]
        label = self.labels[index]
        return picture, label
    
    def __len__(self):
        return len(self.labels)
```

# 分类器预训练

利用原图训练分类器，分类器为ResNet152

# 采样矩阵和恢复矩阵预训练

一个卷积层和一个转置卷积层

```python
class sample(nn.Module):
    def __init__(self):
        super(sample, self).__init__()
        self.k = nn.Conv2d(in_channels=3, out_channels=231, kernel_size=16, stride=16, bias=False) # [3*256*256]->[231*16*16] 0.3倍采样
        self.k_auxiliary = nn.ConvTranspose2d(231, 3, kernel_size=16,  stride=16, bias=False) # [231*16*16]->[3*256*256] 利用转置卷积恢复原图
        
    def forward(self, x):
        out = self.k(x)
        out = self.k_auxiliary(out)
        return out
```

损失函数为原图和恢复图像之间的均方误差损失。

# 整体模型训练

整体模型训练分为三部分，分别为去噪网络训练（去除信道噪声，信道编码），分类器网络微调（去噪完图像的分布与原图分布发生变化，微调分类器最后三层），策略网络训练， 依次进行。其中策略网络的训练分为三个阶段。第一阶段固定信道增益感知网络的参数，优化图像感知网络和决策网络的参数；第二阶段固定图像感知网络的参数，优化其他两个网络的参数；第三阶段同时优化三个网络的参数。

# 感谢

@ARTICLE{
  author={Kang, Xu and Song, Bin and Guo, Jie and Qin, Zhijin and Yu, Fei Richard},
  journal={IEEE Transactions on Communications}, 
  title={Task-Oriented Image Transmission for Scene Classification in Unmanned Aerial Systems}, 
  year={2022},
  volume={70},
  number={8},
  pages={5181-5192},
  doi={10.1109/TCOMM.2022.3182325}}
