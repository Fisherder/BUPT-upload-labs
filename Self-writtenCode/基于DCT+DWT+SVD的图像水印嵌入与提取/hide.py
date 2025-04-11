import numpy as np
import cv2
import pywt
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

watermark_path = r".\images\watermarks\watermark.bmp"
carrier_path = r".\images\photos\bupt.bmp"
alpha = 0.05

watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
W1 = watermark_img
Uw, Sw, Vw = np.linalg.svd(W1, full_matrices=False)
np.save(r".\npys\Uw.npy", Uw)
np.save(r".\npys\Vw.npy", Vw)
np.save(r".\npys\Sw.npy", Sw)

# 读取载体图像
carrier_img = cv2.imread(carrier_path)
carrier_img_yuv = cv2.cvtColor(carrier_img, cv2.COLOR_BGR2YUV)
Y = carrier_img_yuv[:, :, 0]

# 3 级 DWT 分解
LL_Y, (LH_Y, HL_Y, HH_Y) = pywt.dwt2(Y, 'haar')
LL1_Y, (LH1_Y, HL1_Y, HH1_Y) = pywt.dwt2(LL_Y, 'haar')
LL2_Y, (LH2_Y, HL2_Y, HH2_Y) = pywt.dwt2(LL1_Y, 'haar')

# 对 HH2_Y 应用 DCT
HH2_Ydct = dct(dct(HH2_Y, axis=0, norm='ortho'), axis=1, norm='ortho')

# 对 HH2_Ydct 进行 SVD，并嵌入水印
HUw, HSw, HVw = np.linalg.svd(HH2_Ydct, full_matrices=False)
np.save(r".\npys\HSw.npy", HSw)
HSw_truncated = HSw[:len(Sw)]
HSw_truncated_hat = HSw_truncated + alpha * Sw
HSw[:len(Sw)] = HSw_truncated_hat
HH2_Ydcthat = np.dot(HUw, np.dot(np.diag(HSw), HVw))

# 逆变换DCT 
HH2_hat = idct(idct(HH2_Ydcthat, axis=0, norm='ortho'), axis=1, norm='ortho')

# 逐级逆 DWT
LL1_hat = pywt.idwt2((LL2_Y, (LH2_Y, HL2_Y, HH2_hat)), 'haar')
LL_hat = pywt.idwt2((LL1_hat, (LH1_Y, HL1_Y, HH1_Y)), 'haar')
Y_hat = pywt.idwt2((LL_hat, (LH_Y, HL_Y, HH_Y)), 'haar')

# 替换 Y 通道，重构 RGB 图像
carrier_img_yuv[:, :, 0] = Y_hat
watermarked_img = cv2.cvtColor(carrier_img_yuv, cv2.COLOR_YUV2BGR)

# 显示图像
plt.figure(figsize=(12, 8))

# 显示嵌入前的图像
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(carrier_img, cv2.COLOR_BGR2RGB))
plt.title("原始图像", fontsize = 20)
plt.axis('off')

# 显示嵌入后的图像
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB))
plt.title("嵌入图像", fontsize = 20)
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存水印图像
cv2.imwrite(r".\images\photos\buptstegoR.bmp", watermarked_img)