# BUPT-upload-labs

该仓库是信息安全管理课程的小组代码仓库。

现有成员上传功能代码介绍如下。

# 1. 文件内容统计工具

## 项目简介

这是一个用Java实现的简单文件内容统计工具，可以统计文件的行数、单词数和字符数。该工具支持UTF-8编码的文件，并提供用户友好的交互界面。

## 功能说明

1. **统计功能**：
   - 统计文件的行数
   - 统计文件的单词数（以空格、制表符或换行符分隔）
   - 统计文件的字符数（包括空格）

2. **用户交互**：
   - 提示用户输入文件路径
   - 支持输入 `exit` 退出程序
   - 提供详细的错误信息（如文件不存在、路径无效等）

3. **异常处理**：
   - 捕获并处理文件不存在、路径无效等异常
   - 确保程序在异常情况下不会崩溃

## 使用方法

### 1. 编译代码

确保已安装Java开发工具包（JDK），然后在命令行中运行以下命令编译代码：

```bash
javac FileContentAnalyzer.java
```

### 2. 运行程序

编译成功后，运行以下命令启动程序：

```
java FileContentAnalyzer
```

### 3. 输入文件路径

程序启动后，会提示您输入文件路径。例如：

```
请输入文件路径（输入 'exit' 退出程序）：
```

输入文件的完整路径（如 `C:\example.txt` 或 `/home/user/example.txt`），然后按回车键。

### 4. 查看统计结果

程序会输出文件的行数、单词数和字符数。例如：

```
统计结果：
文件路径: C:\example.txt
行数: 10
单词数: 50
字符数: 300
```

### 5. 退出程序

输入 `exit` 并按回车键即可退出程序。

## 示例输出

```
欢迎使用文件内容统计工具
请输入文件路径（输入 'exit' 退出程序）：
C:\example.txt

统计结果：
文件路径: C:\example.txt
行数: 10
单词数: 50
字符数: 300

请输入下一个文件路径（输入 'exit' 退出程序）：
```

## 依赖项

- Java Development Kit (JDK) 8 或更高版本



# 2. 基于生成对抗网络(GAN)的动漫头像生成系统

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

    debug_file='/tmp/debuggan' # 存在该文件则进入debug模式
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

### 兼容性测试
train 
- [x] GPU  
- [ ] CPU  
- [ ] Python2
- [x] Python3

test: 

- [x] GPU
- [x] CPU
- [ ] Python2
- [x] Python3



# 3. BayesSpamEmail

一个使用 Python 实现的基于贝叶斯分类的简单垃圾邮件识别器。

测试集共 400 封邮件（正常邮件与垃圾邮件各 200 封），分类准确率达到 **95.15%**。即使仅统计词频计算概率，分类效果仍然表现良好。

## 1. 环境准备

- Python 3.4+
- 结巴分词工具：[https://github.com/fxsjy/jieba](https://github.com/fxsjy/jieba)

## 2. 贝叶斯公式

目标是根据词向量 $w = (w_1, w_2, \dots, w_n)$ 计算该邮件是垃圾邮件的概率 $P(s \mid w)$，其中 $s$ 表示邮件为垃圾邮件。

根据贝叶斯定理和全概率公式，有：

$$
P(s \mid w_1, w_2, \dots, w_n) = \frac{P(w_1, w_2, \dots, w_n \mid s) P(s)}{P(w_1, w_2, \dots, w_n)}
$$

由于：

$$
P(w_1, w_2, \dots, w_n) = P(w_1, w_2, \dots, w_n \mid s) \cdot P(s) + P(w_1, w_2, \dots, w_n \mid s') \cdot P(s')
$$

并且采用朴素贝叶斯条件独立假设以及先验概率 $P(s) = P(s') = 0.5$，可简化为：

$$
P(s \mid w) = \frac{\prod_{j=1}^n P(w_j \mid s)}{\prod_{j=1}^n P(w_j \mid s) + \prod_{j=1}^n P(w_j \mid s')}
$$

进一步利用贝叶斯公式 $P(w_j \mid s) = \frac{P(s \mid w_j) \cdot P(w_j)}{P(s)}$，可以将其表示为：

$$
P(s \mid w) = \frac{\prod_{j=1}^n P(s \mid w_j)}{\prod_{j=1}^n P(s \mid w_j) + \prod_{j=1}^n (1 - P(s \mid w_j))}
$$

> 我们使用上式计算 $P(s \mid w)$，因为它仅依赖于 $P(s \mid w_j)$，简化了计算。

## 3. 实现步骤

实现代码已提供，以下是主要思路：

1. **分词与清洗**  
   对训练集邮件使用结巴分词，结合停用词表进行过滤，并通过正则表达式去除非中文字符。

2. **构建词典**  
   分别统计正常邮件与垃圾邮件中各个词的出现频次，构建两个词典。  
   例如：词“疯狂”在 8000 封正常邮件中出现 20 次，在 8000 封垃圾邮件中出现 200 次。

3. **处理测试集**  
   对测试邮件执行相同处理，并找出使 $P(s \mid w)$ 最大的 15 个关键词。  
   - 若词只出现在垃圾邮件中，设 $P(w \mid s') = 0.01$；反之亦然。  
   - 若词在两个词典中都未出现，则设 $P(s \mid w) = 0.4$。

4. **分类判定**  
   对每封邮件选出的 15 个关键词利用前述公式计算 $P(s \mid w)$，若该值大于阈值 $\alpha$（通常设为 0.9），则判断为垃圾邮件，否则为正常邮件。

---



# 4. 基于DCT+DWT+SVD的图像水印嵌入与提取

一个使用python实现的基于`DCT+DWT+SVD`的图像水印嵌入与提取。使用嵌入后图片与原始图片的`psnr`值来判断本方法的隐蔽性；使用经过高斯噪声攻击后提取的水印与原始水印的`NC`值来判断本方法的鲁棒性。

## 1.原理介绍

### 1.1 离散余弦变换（DCT）

DCT将图像从空间域转换到频域，通过余弦基函数的正交特性分解图像。具体步骤包括：

- 分块处理：将图像分割为8×8或16×16的小块（如JPEG标准）
- 频域转换：每个图像块通过二维DCT变换得到系数矩阵，其中低频系数（靠近左上角）集中了大部分能量，高频系数（右下角）对应细节和噪声

### 1.2 离散小波变换（DWT）

DWT通过多级分解将图像划分为不同频率的子带：

- 多分辨率分析：每级分解产生低频（LL）和多个高频子带（LH、HL、HH），低频保留主体信息，高频包含细节
- 方向性：高频子带分别对应水平、垂直和对角线方向的细节

### 1.3 奇异值分解（SVD）

SVD将图像矩阵分解为三个矩阵的乘积:

$$
\begin{equation}
A = U \begin{bmatrix}
\sigma_1 & 0 & \cdots & 0 \\
0 & \sigma_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma_r
\end{bmatrix} V^{\top} \nonumber
\end{equation}
$$

- 特征提取：奇异值（Σ对角线元素）反映图像能量分布，大奇异值对应主要结构，小值对应细节或噪声
- 稳定性：对几何攻击（旋转、缩放）具有鲁棒性，适合水印和压缩

## 2.实现步骤

实现代码已提供，以下主要介绍实现思路。

### 2.1 嵌入水印

- 对图像YUV三通道中的Y通道进行三级DWT变换（使用LL子带）
- 对HH2（三级变换后的HH子带）进行DCT变换
- 分别对水印图像和HH2_dct（DCT变换后的HH2）进行SVD得到奇异值SW和HSW
- 对HSW进行替换：

$$
HSW'=HSW+\alpha*SW(\alpha为嵌入因子)
$$

- 对替换后的HSW进行DCT逆变换及三级DWT逆变换得到替换后的Y通道
- 使用替换后的Y通道替换原始图像的Y通道，并变换为RGB通道得到嵌入水印后的图像

### 2.2 提取水印

- 对嵌入水印后的图像YUV三通道中的Y通道进行三级DWT变换（使用LL子带）
- 对HH2（三级变换后的HH子带）进行DCT变换得到HH2_Ydct_watermarked
- 对水印图像进行SVD得到UW、VW
- 对HH2_Ydct_watermarked进行SVD的到HSW'
- 使用HSW'和HSW得到SW：
  
$$
SW=\frac{HSW'-HSW}{\alpha}
$$

- 使用UW、SW、VW得到水印图像

### 2.3 攻击图像

- 对嵌入水印后的图像增加随机产生的噪声

### 2.4 PSNR值计算

- 对原始图像及嵌入水印后的图像进行`psnr`值的计算



# 5. SM2_Crypto

这段代码是使用sm-crypto-v2库进行SM2加密、解密、签名和验签的测试。



# 6. AES and SM4 Crypto

代码支持了 AES 和 SM4 两种加密算法。并且在项目中提供了简单易用的 Java API，方便开发者在应用程序中集成加密功能。

### 功能模块

#### 1. AES 加密模块（`AesCryptoUtils.java`）
AES（高级加密标准）是一种对称加密算法，广泛应用于数据加密。本模块提供了基于 AES 的加密和解密功能，支持 CBC 模式和 PKCS5Padding 填充方式。

- **加密方法**
  
  ```java
  public static String encrypt(String plainText, String key, String iv) throws Exception
  ```
  - **参数**
    - `plainText`：明文字符串，需要被加密的数据。
    - `key`：加密密钥，长度必须为 16 字节（128 位）。
    - `iv`：初始化向量（可选），长度必须为 16 字节。如果未提供，则使用默认的全零向量。
  - **返回值**
    - 加密后的密文，以 Base64 编码的字符串形式返回。
  
- **解密方法**
  ```java
  public static String decrypt(String encryptedText, String key, String iv) throws Exception
  ```
  - **参数**
    - `encryptedText`：密文字符串，需要被解密的数据。
    - `key`：解密密钥，长度必须为 16 字节（128 位）。
    - `iv`：初始化向量（可选），长度必须为 16 字节。如果未提供，则使用默认的全零向量。
  - **返回值**
    - 解密后的明文字符串。

- **密钥和初始化向量校验**
  ```java
  private static void validateKeyIV(byte[] key, byte[] iv)
  ```
  - **功能**
    - 校验密钥和初始化向量的长度是否符合要求。AES 密钥长度必须为 16 字节，初始化向量长度也必须为 16 字节。

#### 2. SM4 加密模块（`Sm4CryptoUtils.java`）
SM4 是我国自主设计的分组密码算法，具有高效性和安全性。本模块提供了基于 SM4 的加密和解密功能，支持 CBC 模式和 PKCS5Padding 填充方式。

- **加密方法**
  ```java
  public static String encrypt(String plainText, String key, String iv) throws Exception
  ```
  - **参数**
    - `plainText`：明文字符串，需要被加密的数据。
    - `key`：加密密钥，长度必须为 16 字节（128 位）。
    - `iv`：初始化向量（可选），长度必须为 16 字节。如果未提供，则使用默认的全零向量。
  - **返回值**
    - 加密后的密文，以 Base64 编码的字符串形式返回。

- **解密方法**
  ```java
  public static String decrypt(String encryptedText, String key, String iv) throws Exception
  ```
  - **参数**
    - `encryptedText`：密文字符串，需要被解密的数据。
    - `key`：解密密钥，长度必须为 16 字节（128 位）。
    - `iv`：初始化向量（可选），长度必须为 16 字节。如果未提供，则使用默认的全零向量。
  - **返回值**
    - 解密后的明文字符串。

- **密钥和初始化向量校验**
  ```java
  private static void validateKeyIV(byte[] key, byte[] iv)
  ```
  - **功能**
    - 校验密钥和初始化向量的长度是否符合要求。SM4 密钥长度必须为 16 字节，初始化向量长度也必须为 16 字节。

### 使用示例

#### AES 加密解密示例
```java
import com.example.cryptotool.utils.AesCryptoUtils;

public class AesExample {
    public static void main(String[] args) {
        try {
            String plainText = "Hello, AES!";
            String key = "0123456789abcdef"; // 16 字节密钥
            String iv = "fedcba9876543210"; // 16 字节初始化向量

            // 加密
            String encryptedText = AesCryptoUtils.encrypt(plainText, key, iv);
            System.out.println("Encrypted Text: " + encryptedText);

            // 解密
            String decryptedText = AesCryptoUtils.decrypt(encryptedText, key, iv);
            System.out.println("Decrypted Text: " + decryptedText);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

#### SM4 加密解密示例
```java
import com.example.cryptotool.utils.Sm4CryptoUtils;

public class Sm4Example {
    public static void main(String[] args) {
        try {
            String plainText = "Hello, SM4!";
            String key = "0123456789abcdef"; // 16 字节密钥
            String iv = "fedcba9876543210"; // 16 字节初始化向量

            // 加密
            String encryptedText = Sm4CryptoUtils.encrypt(plainText, key, iv);
            System.out.println("Encrypted Text: " + encryptedText);

            // 解密
            String decryptedText = Sm4CryptoUtils.decrypt(encryptedText, key, iv);
            System.out.println("Decrypted Text: " + decryptedText);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### 依赖说明
代码依赖于 Bouncy Castle 提供的加密库，用于支持 SM4 算法。在使用 SM4 功能之前，请确保已添加 Bouncy Castle 提供者。

```java
Security.addProvider(new BouncyCastleProvider());
```

### 注意事项
1. **密钥和初始化向量长度**：AES 和 SM4 的密钥长度必须为 16 字节（128 位），初始化向量长度也必须为 16 字节。
2. **安全性**：请妥善保管密钥和初始化向量，避免泄露。
3. **异常处理**：在实际使用中，请对加密和解密操作进行异常处理，确保程序的健壮性。
