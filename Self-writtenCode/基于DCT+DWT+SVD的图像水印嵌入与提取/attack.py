import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def gaussian_attack(img, mean=0, sigma=0.1):
    """
    对输入图像添加高斯噪声
    
    参数:
        img: 输入图像，numpy数组格式
        mean: 高斯噪声的均值，默认为0
        sigma: 高斯噪声的标准差，默认为0.1
        
    返回:
        添加噪声后的图像
    """
    # 检查输入类型
    if not isinstance(img, np.ndarray):
        raise TypeError("输入必须是numpy数组")
    
    # 转换为浮点型并归一化
    img_float = img.astype(np.float32) / 255
    
    # 添加高斯噪声
    noise = np.random.normal(mean, sigma, img_float.shape)
    img_gaussian = img_float + noise
    
    # 裁剪值到有效范围并转回uint8
    img_gaussian = np.clip(img_gaussian, 0, 1)
    img_gaussian = np.uint8(img_gaussian * 255)
    
    return img_gaussian

def process_image(input_path, output_path, mean=0, sigma=0.1, show=False):
    """
    处理图像并添加高斯噪声
    
    参数:
        input_path: 输入图像路径
        output_path: 输出图像路径
        mean: 高斯噪声均值
        sigma: 高斯噪声标准差
        show: 是否显示处理后的图像
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图像: {input_path}，可能是不支持的格式")
    
    # 应用高斯噪声
    attacked_img = gaussian_attack(img, mean, sigma)
    
    # 保存结果
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not cv2.imwrite(output_path, attacked_img):
        raise IOError(f"无法写入图像: {output_path}")
    
    print(f"已成功处理图像并保存到: {output_path}")
    
    # 可选：显示图像
    if show:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(os.path.basename(input_path))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(attacked_img, cv2.COLOR_BGR2RGB))
        plt.title(os.path.basename(output_path))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.close()  # 关闭图形窗口释放资源

def main():
    parser = argparse.ArgumentParser(description='对图像添加高斯噪声')
    parser.add_argument('--input', '-i', required=True, help='输入图像路径')
    parser.add_argument('--output', '-o', required=True, help='输出图像路径')
    parser.add_argument('--mean', '-m', type=float, default=0, help='高斯噪声均值')
    parser.add_argument('--sigma', '-s', type=float, default=0.1, help='高斯噪声标准差')
    parser.add_argument('--show', '-v', action='store_true', help='显示处理后的图像')
    
    args = parser.parse_args()
    
    try:
        process_image(args.input, args.output, args.mean, args.sigma, args.show)
    except Exception as e:
        print(f"处理图像时出错: {e}")

if __name__ == "__main__":
    main()
