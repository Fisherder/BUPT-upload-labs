import numpy as np
import cv2
import pywt
from scipy.fftpack import dct
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 归一化相关性计算函数
def NC(template, img):
    template = template.astype(np.uint8)
    img = img.astype(np.uint8)
    result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)[0][0]
    return result

watermark_img_path = r".\images\watermarks\watermark.bmp"
watermark_img = cv2.imread(watermark_img_path, cv2.IMREAD_GRAYSCALE)

# 加载相应数据
Uw = np.load("Uw.npy")
Sw = np.load("Sw.npy")
Vw = np.load("Vw.npy")
HSw = np.load("HSw.npy")
alpha = 0.05

# 读取含有水印的图像
watermarked_img_path =r".\images\photos\buptstegoR.bmp"
watermarked_img = cv2.imread(watermarked_img_path)
watermarked_img_yuv = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YUV)
Y_watermarked = watermarked_img_yuv[:, :, 0]

# 3 级 DWT 分解（含水印图像）
LL_Y_watermarked, (LH_Y_watermarked, HL_Y_watermarked, HH_Y_watermarked) = pywt.dwt2(Y_watermarked, 'haar')
LL1_Y_watermarked, (LH1_Y_watermarked, HL1_Y_watermarked, HH1_Y_watermarked) = pywt.dwt2(LL_Y_watermarked, 'haar')
LL2_Y_watermarked, (LH2_Y_watermarked, HL2_Y_watermarked, HH2_Y_watermarked) = pywt.dwt2(LL1_Y_watermarked, 'haar')

# 对 HH2_Y_watermarked 应用 DCT
HH2_Ydct_watermarked = dct(dct(HH2_Y_watermarked, axis=0, norm='ortho'), axis=1, norm='ortho')

# 对水印图像的 DCT 变换后的 HH2_Y 进行 SVD
HUw_watermarked, HSw_watermarked, HVw_watermarked = np.linalg.svd(HH2_Ydct_watermarked, full_matrices=False)

# 提取水印的奇异值
watermark_extracted = (HSw_watermarked[:len(Sw)] - HSw[:len(Sw)]) / alpha

# 水印恢复
watermark_extracted = np.dot(Uw, np.dot(np.diag(watermark_extracted), Vw))
watermark_extracted = np.clip(watermark_extracted, 0, 255).astype(np.uint8)


print("original:\n",watermark_img)
print("watermark_extracted;\n",watermark_extracted)

# 进行二值化处理，恢复原始的二值水印
#_, watermark_extracted_bin = cv2.threshold(watermark_extracted, 128, 255, cv2.THRESH_BINARY)

# 显示图像
plt.figure(figsize=(12, 8))

# 显示原始水印
plt.subplot(1, 2, 1)
plt.imshow(watermark_img, cmap="gray")
plt.title("原始水印")
plt.axis('off')

# 显示未遭受攻击水印
plt.subplot(1, 2, 2)
plt.imshow(watermark_extracted, cmap="gray")
plt.title("攻击后水印")
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存提取出的水印
cv2.imwrite(r".\images\watermarks\watermark2.bmp", watermark_extracted)

# 计算归一化相关性
nc = NC(watermark_extracted, watermark_img)
print(f"NC: {nc}")
