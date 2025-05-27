import os
import sys
import numpy as np
import cv2
import pywt
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def setup_logger():
    """配置日志记录器"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

def validate_path(path, is_input=True, create_dir=False):
    """验证文件/目录路径有效性"""
    path_obj = Path(path)
    
    if is_input:
        if not path_obj.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        if path_obj.is_dir():
            raise ValueError(f"预期文件但得到目录: {path}")
    else:
        if create_dir:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        if path_obj.exists() and path_obj.is_dir():
            raise ValueError(f"输出路径是目录: {path}")
    
    return str(path)

def load_image(image_path, grayscale=False):
    """安全加载图像文件"""
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(image_path, flags)
    
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    return img

def save_image(image, output_path):
    """安全保存图像文件"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"无法保存图像: {output_path}")
    
    logger.info(f"图像已保存至: {output_path}")

def load_npy_data(file_path):
    """安全加载numpy数据文件"""
    try:
        return np.load(file_path)
    except Exception as e:
        raise ValueError(f"无法加载数据文件 {file_path}: {str(e)}")

def NC(template, img):
    """计算归一化相关性"""
    template = template.astype(np.float32)
    img = img.astype(np.float32)
    
    # 确保图像尺寸一致
    if template.shape != img.shape:
        logger.warning(f"模板和图像尺寸不一致: {template.shape} vs {img.shape}")
        min_height = min(template.shape[0], img.shape[0])
        min_width = min(template.shape[1], img.shape[1])
        template = template[:min_height, :min_width]
        img = img[:min_height, :min_width]
    
    result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)[0][0]
    return result

def extract_watermark(watermarked_img, Uw, Sw, Vw, HSw, alpha=0.05, wavelet_level=3, wavelet_type='haar'):
    """
    从含水印图像中提取水印
    
    参数:
        watermarked_img: 含水印的图像
        Uw, Sw, Vw, HSw: 水印的SVD分解参数
        alpha: 水印强度参数
        wavelet_level: 小波变换级数
        wavelet_type: 小波变换类型
    
    返回:
        extracted_watermark: 提取出的水印图像
    """
    # 转换为YUV颜色空间并提取亮度通道
    watermarked_img_yuv = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YUV)
    Y_watermarked = watermarked_img_yuv[:, :, 0]
    
    # 多级小波分解
    LL = Y_watermarked
    for _ in range(wavelet_level - 1):
        LL, (LH, HL, HH) = pywt.dwt2(LL, wavelet_type)
    
    # 最后一级分解
    LL, (LH, HL, HH) = pywt.dwt2(LL, wavelet_type)
    
    # 对HH子带应用DCT
    HH_dct = dct(dct(HH, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    # 对DCT变换后的HH子带进行SVD
    HU, HS, HV = np.linalg.svd(HH_dct, full_matrices=False)
    
    # 提取水印的奇异值
    min_len = min(len(HS), len(Sw))
    watermark_extracted_singular = (HS[:min_len] - HSw[:min_len]) / max(alpha, 1e-10)  # 避免除零
    
    # 水印恢复
    watermark_extracted = np.dot(Uw, np.dot(np.diag(watermark_extracted_singular), Vw))
    watermark_extracted = np.clip(watermark_extracted, 0, 255).astype(np.uint8)
    
    return watermark_extracted

def process_watermark_extraction(
    watermarked_img_path, 
    watermark_img_path,
    output_path,
    uw_path, sw_path, vw_path, hsw_path,
    alpha=0.05,
    show_results=False,
    threshold=128
):
    """
    处理水印提取的完整流程
    
    参数:
        watermarked_img_path: 含水印图像路径
        watermark_img_path: 原始水印图像路径
        output_path: 输出提取水印的路径
        uw_path, sw_path, vw_path, hsw_path: 水印SVD参数路径
        alpha: 水印强度参数
        show_results: 是否显示结果
        threshold: 二值化阈值
    """
    try:
        # 验证路径
        watermarked_img_path = validate_path(watermarked_img_path)
        watermark_img_path = validate_path(watermark_img_path)
        output_path = validate_path(output_path, is_input=False, create_dir=True)
        
        # 加载数据
        logger.info("加载图像和数据文件...")
        watermarked_img = load_image(watermarked_img_path)
        original_watermark = load_image(watermark_img_path, grayscale=True)
        
        Uw = load_npy_data(uw_path)
        Sw = load_npy_data(sw_path)
        Vw = load_npy_data(vw_path)
        HSw = load_npy_data(hsw_path)
        
        # 提取水印
        logger.info("正在提取水印...")
        extracted_watermark = extract_watermark(watermarked_img, Uw, Sw, Vw, HSw, alpha)
        
        # 二值化处理
        _, binary_watermark = cv2.threshold(extracted_watermark, threshold, 255, cv2.THRESH_BINARY)
        
        # 保存结果
        save_image(binary_watermark, output_path)
        
        # 计算NC值
        nc_value = NC(binary_watermark, original_watermark)
        logger.info(f"归一化相关性(NC): {nc_value:.4f}")
        
        # 显示结果
        if show_results:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original_watermark, cmap="gray")
            plt.title("原始水印")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(extracted_watermark, cmap="gray")
            plt.title("提取的水印(灰度)")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(binary_watermark, cmap="gray")
            plt.title("提取的水印(二值化)")
            plt.axis('off')
            
            plt.tight_layout()
            plt.suptitle(f"水印提取结果 (NC: {nc_value:.4f})", fontsize=16)
            plt.show()
            plt.close()
        
        return binary_watermark, nc_value
    
    except Exception as e:
        logger.error(f"水印提取过程中出错: {str(e)}")
        raise

def main():
    """主函数，处理命令行参数并执行水印提取"""
    parser = argparse.ArgumentParser(description='数字水印提取工具')
    
    # 必选参数
    parser.add_argument('--watermarked', '-w', required=True, help='含水印图像路径')
    parser.add_argument('--original', '-o', required=True, help='原始水印图像路径')
    parser.add_argument('--output', '-p', required=True, help='输出提取水印的路径')
    parser.add_argument('--uw', required=True, help='Uw.npy 文件路径')
    parser.add_argument('--sw', required=True, help='Sw.npy 文件路径')
    parser.add_argument('--vw', required=True, help='Vw.npy 文件路径')
    parser.add_argument('--hsw', required=True, help='HSw.npy 文件路径')
    
    # 可选参数
    parser.add_argument('--alpha', '-a', type=float, default=0.05, help='水印强度参数 (默认: 0.05)')
    parser.add_argument('--show', '-s', action='store_true', help='显示处理结果')
    parser.add_argument('--threshold', '-t', type=int, default=128, help='二值化阈值 (默认: 128)')
    
    args = parser.parse_args()
    
    try:
        process_watermark_extraction(
            args.watermarked,
            args.original,
            args.output,
            args.uw,
            args.sw,
            args.vw,
            args.hsw,
            args.alpha,
            args.show,
            args.threshold
        )
        logger.info("水印提取完成")
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()