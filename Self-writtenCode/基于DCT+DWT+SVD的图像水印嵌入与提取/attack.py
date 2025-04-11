import cv2
import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def gaussian_attack(img, mean, sigma):
    img = img.astype(np.float32) / 255
    noise = np.random.normal(mean, sigma, img.shape)
    img_gaussian = img + noise
    img_gaussian = np.clip(img_gaussian, 0, 1)
    img_gaussian = np.uint8(img_gaussian * 255)
    return img_gaussian

img_path = r".\images\photos\buptstegoR.bmp"
img = cv2.imread(img_path)

attacked_img = gaussian_attack(img, 0, 0.1)

# 显示图像
plt.figure(figsize=(12, 8))

# 显示原始水印
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("buptstegoR.bmp")
plt.axis('off')

# 显示未遭受攻击水印
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(attacked_img, cv2.COLOR_BGR2RGB))
plt.title("buptstegoR1.bmp")
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存结果
cv2.imwrite("buptstegoR1.bmp", attacked_img)