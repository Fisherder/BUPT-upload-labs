# 基于生成对抗网络(GAN)的动漫头像生成系统

使用PyTorch框架实现的DCGAN架构，包含生成器(NetG)和判别器(NetD)两个神经网络，通过对抗训练学习生成逼真的动漫风格头像。

## 核心代码功能介绍

### 1. 配置系统 (Config类)
- 集中管理所有训练参数
- 支持动态修改数据路径、网络结构、训练超参等
- 包含可视化/调试开关

### 2. 训练系统 (train函数)
- **数据处理流程**：
  - 自动加载数据集
  - 图片中心裁剪→尺寸归一化→像素值归一化
- **对抗训练**：
  - 判别器训练：区分真实/生成图片
  - 生成器训练：生成更逼真的图片
- **训练监控**：
  - Visdom实时可视化损失曲线
  - 定期保存生成样例图片

### 3. 生成系统 (generate函数)
- 加载预训练模型
- 批量生成候选图片(默认512张)
- 使用判别器自动筛选最佳结果(默认保留64张)
- 保存高质量生成结果

### 4. 模型架构
- **NetG生成器**：
  - 输入：100维随机噪声
  - 输出：96x96 RGB动漫头像
  - 结构：反卷积神经网络
- **NetD判别器**：
  - 输入：96x96 RGB图片
  - 输出：图片真实性概率
  - 结构：卷积神经网络


## 环境准备

- 本程序需要安装[PyTorch](https://pytorch.org/)
- 还需要通过`pip install -r requirements.txt` 安装其它依赖


## 数据准备

更好的图片生成效果更好

- 可以自己写爬虫爬取[Danbooru](http://link.zhihu.com/?target=http%3A//safebooru.donmai.us/)或者[konachan](http://konachan.net/)
- 如果你不想从头开始爬图片，可以直接使用爬好的头像数据（275M，约5万多张图片）：https://pan.baidu.com/s/1eSifHcA 提取码：g5qa
感谢知乎用户[何之源](https://www.zhihu.com/people/he-zhi-yuan-16)爬取的数据。
请把所有的图片保存于data/face/目录下，形如
```
data/
└── faces/
    ├── 0000fdee4208b8b7e12074c920bc6166-0.jpg
    ├── 0001a0fca4e9d2193afea712421693be-0.jpg
    ├── 0001d9ed32d932d298e1ff9cc5b7a2ab-0.jpg
    ├── 0001d9ed32d932d298e1ff9cc5b7a2ab-1.jpg
    ├── 00028d3882ec183e0f55ff29827527d3-0.jpg
    ├── 00028d3882ec183e0f55ff29827527d3-1.jpg
    ├── 000333906d04217408bb0d501f298448-0.jpg
    ├── 0005027ac1dcc32835a37be806f226cb-0.jpg
```
即data目录下只有一个文件夹，文件夹中有所有的图片

## 用法
如果想要使用visdom可视化，请先运行`python2 -m visdom.server`启动visdom服务
基本用法：
```
Usage： python main.py FUNCTION --key=value,--key2=value2 ..
```

- 训练
```bash
python main.py train --gpu --vis=False
```

- 生成图片

[点此](https://yun.sfo2.digitaloceanspaces.com/pytorch_book/pytorch_book/netg_200.pth)可下载预训练好的生成模型，如果想要下载预训练的判别模型，请[点此](https://yun.sfo2.digitaloceanspaces.com/pytorch_book/pytorch_book/netd_200.pth)
```bash
python main.py generate --nogpu --vis=False \
            --netd-path=checkpoints/netd_200.pth \
            --netg-path=checkpoints/netg_200.pth \
            --gen-img=result.png \
            --gen-num=64
```
完整的选项及默认值
```python
    data_path = 'data/' # 数据集存放路径
    num_workers = 4 # 多进程加载数据所用的进程数
    image_size = 96 # 图片尺寸
    batch_size = 256
    max_epoch =  200
    lr1 = 2e-4 # 生成器的学习率
    lr2 = 2e-4 # 判别器的学习率
    beta1=0.5 # Adam优化器的beta1参数
    gpu=True # 是否使用GPU --nogpu或者--gpu=False不使用gpu
    nz=100 # 噪声维度
    ngf = 64 # 生成器feature map数
    ndf = 64 # 判别器feature map数
    
    save_path = 'imgs/' #训练时生成图片保存路径
    
    vis = True # 是否使用visdom可视化
    env = 'GAN' # visdom的env
    plot_every = 20 # 每间隔20 batch，visdom画图一次

    debug_file='./debug_gan.flag' # 存在该文件则进入debug模式（在项目目录下创建debug_gan.flag文件）
    d_every=1 # 每1个batch训练一次判别器
    g_every=5 # 每5个batch训练一次生成器
    decay_every=10 # 没10个epoch保存一次模型
    netd_path = 'checkpoints/netd_211.pth' #预训练模型
    netg_path = 'checkpoints/netg_211.pth'
    
    # 只测试不训练
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64 
    gen_search_num = 512 
    gen_mean = 0 # 噪声的均值
    gen_std = 1 #噪声的方差
   
```
生成的部分图片：
![imgs](imgs/gan-results.png)


### 兼容性测试
train 
- [x] GPU  
- [] CPU  
- [] Python2
- [x] Python3

test: 

- [x] GPU
- [x] CPU
- [] Python2
- [x] Python3
