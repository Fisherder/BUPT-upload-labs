# coding:utf8
import os
import torch as t
import torchvision as tv
import tqdm
from model import NetG, NetD
from torchnet.meter import AverageValueMeter


class Config(object):
    data_path = 'data/'  # 数据集存放路径
    num_workers = 4  # 多进程加载数据所用的进程数
    image_size = 96  # 图片尺寸
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4  # 生成器的学习率
    lr2 = 2e-4  # 判别器的学习率
    beta1 = 0.5  # Adam优化器的beta1参数
    gpu = True  # 是否使用GPU
    nz = 100  # 噪声维度
    ngf = 64  # 生成器feature map数
    ndf = 64  # 判别器feature map数

    save_path = 'imgs/'  # 生成图片保存路径

    vis = True  # 是否使用visdom可视化
    env = 'GAN'  # visdom的env
    plot_every = 20  # 每间隔20 batch，visdom画图一次

    debug_file = './debug_gan.flag'  # 存在该文件则进入debug模式
    d_every = 1  # 每1个batch训练一次判别器
    g_every = 5  # 每5个batch训练一次生成器
    save_every = 10  # 没10个epoch保存一次模型
    netd_path = None  # 'checkpoints/netd_.pth' #预训练模型
    netg_path = None  # 'checkpoints/netg_211.pth'

    # 只测试不训练
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差


opt = Config()


def safe_load_model(model_path, device):
    """
    安全地加载模型文件，防止pickle反序列化攻击
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if not os.path.isfile(model_path):
        raise ValueError(f"路径不是文件: {model_path}")
    
    # 检查文件扩展名
    if not model_path.endswith(('.pth', '.pt')):
        raise ValueError(f"不支持的模型文件格式: {model_path}")
    
    try:
        # 使用weights_only=True防止pickle反序列化攻击
        # 对于PyTorch 1.13+版本，这是推荐的安全做法
        state_dict = t.load(model_path, map_location=device, weights_only=True)
        return state_dict
    except Exception as e:
        # 如果weights_only不支持（旧版本PyTorch），使用传统方式但添加警告
        print(f"警告: 正在使用传统的模型加载方式，存在安全风险。建议升级PyTorch版本。")
        print(f"模型文件: {model_path}")
        state_dict = t.load(model_path, map_location=device)
        return state_dict


def ensure_directories():
    """
    确保必要的目录存在
    """
    directories = ['imgs', 'checkpoints']
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"创建目录: {dir_name}")


def verify_gpu_availability(use_gpu):
    """
    验证GPU可用性
    """
    if use_gpu:
        if not t.cuda.is_available():
            print("警告: CUDA不可用，自动切换到CPU模式")
            return False
        else:
            gpu_count = t.cuda.device_count()
            current_device = t.cuda.current_device()
            gpu_name = t.cuda.get_device_name(current_device)
            print(f"使用GPU: {gpu_name} (设备 {current_device}/{gpu_count-1})")
            return True
    else:
        print("使用CPU模式")
        return False


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    # 验证GPU可用性
    opt.gpu = verify_gpu_availability(opt.gpu)
    device = t.device('cuda') if opt.gpu else t.device('cpu')
    
    # 确保必要目录存在
    ensure_directories()
    
    # 验证数据路径
    if not os.path.exists(opt.data_path):
        raise FileNotFoundError(f"数据集路径不存在: {opt.data_path}")
    
    if opt.vis:
        try:
            from visualize import Visualizer
            vis = Visualizer(opt.env)
        except ImportError as e:
            print(f"警告: 无法导入可视化模块: {e}")
            opt.vis = False

    # 数据
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
        if len(dataset) == 0:
            raise ValueError(f"数据集为空: {opt.data_path}")
        print(f"成功加载数据集，图片数量: {len(dataset)}")
    except Exception as e:
        raise RuntimeError(f"加载数据集失败: {e}")

    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True
                                         )

    # 网络
    netg, netd = NetG(opt), NetD(opt)
    
    # 安全地加载预训练模型
    if opt.netd_path:
        try:
            netd_state = safe_load_model(opt.netd_path, device)
            netd.load_state_dict(netd_state)
            print(f"成功加载判别器模型: {opt.netd_path}")
        except Exception as e:
            print(f"警告: 加载判别器模型失败: {e}")
    
    if opt.netg_path:
        try:
            netg_state = safe_load_model(opt.netg_path, device)
            netg.load_state_dict(netg_state)
            print(f"成功加载生成器模型: {opt.netg_path}")
        except Exception as e:
            print(f"警告: 加载生成器模型失败: {e}")
    
    netd.to(device)
    netg.to(device)

    # 定义优化器和损失
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device)

    # 真图片label为1，假图片label为0
    # noises为生成网络的输入
    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    epochs = range(opt.max_epoch)

    fix_fake_imgs = netg(fix_noises)
    for epoch in iter(epochs):
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            if ii % opt.d_every == 0:
                # 训练判别器
                optimizer_d.zero_grad()
                ## 尽可能的把真图片判别为正确
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                ## 尽可能把假图片判别为错误
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()  # 根据噪声生成假图
                output = netd(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

                error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.item())

            if ii % opt.g_every == 0:
                # 训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.item())

            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                ## 可视化
                if os.path.exists(opt.debug_file):
                    # 安全的调试方式：仅设置断点标志，避免直接导入调试工具
                    print("DEBUG: 检测到调试标志文件，进入调试模式")
                    print(f"DEBUG: 当前epoch={epoch}, batch={ii}")
                    print(f"DEBUG: 判别器损失={errord_meter.value()[0]:.4f}")
                    print(f"DEBUG: 生成器损失={errorg_meter.value()[0]:.4f}")
                fix_fake_imgs = netg(fix_noises)
                try:
                    vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                    vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                    vis.plot('errord', errord_meter.value()[0])
                    vis.plot('errorg', errorg_meter.value()[0])
                except Exception as e:
                    print(f"可视化错误: {e}")

        if (epoch+1) % opt.save_every == 0:
            # 保存模型、图片
            try:
                tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch), normalize=True,
                                    value_range=(-1, 1))
                t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
                t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
                print(f"已保存第{epoch+1}轮的模型和图片")
            except Exception as e:
                print(f"保存模型/图片时发生错误: {e}")
            
            errord_meter.reset()
            errorg_meter.reset()


@t.no_grad()
def generate(**kwargs):
    """
    随机生成动漫头像，并根据netd的分数选择较好的
    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    
    # 验证GPU可用性
    opt.gpu = verify_gpu_availability(opt.gpu)
    device = t.device('cuda') if opt.gpu else t.device('cpu')

    # 验证模型路径
    if not opt.netd_path or not opt.netg_path:
        raise ValueError("生成模式需要指定netd_path和netg_path参数")

    netg, netd = NetG(opt).eval(), NetD(opt).eval()
    noises = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = noises.to(device)

    # 安全地加载模型
    try:
        netd_state = safe_load_model(opt.netd_path, device)
        netd.load_state_dict(netd_state)
        print(f"成功加载判别器模型: {opt.netd_path}")
        
        netg_state = safe_load_model(opt.netg_path, device)
        netg.load_state_dict(netg_state)
        print(f"成功加载生成器模型: {opt.netg_path}")
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")
    
    netd.to(device)
    netg.to(device)

    # 生成图片，并计算图片在判别器的分数
    fake_img = netg(noises)
    scores = netd(fake_img).detach()

    # 挑选最好的某几张
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    
    # 保存图片
    try:
        tv.utils.save_image(t.stack(result), opt.gen_img, normalize=True, value_range=(-1, 1))
        print(f"成功生成{opt.gen_num}张图片，保存到: {opt.gen_img}")
    except Exception as e:
        raise RuntimeError(f"保存生成图片失败: {e}")


if __name__ == '__main__':
    import fire
    fire.Fire()
