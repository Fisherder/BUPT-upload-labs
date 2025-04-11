import math
import numpy as np
import cv2

def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return "100.000"
    PIXEL_MAX = 255.0
    psnr_value = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return f"{psnr_value:.3f}"


original = cv2.imread(r'.\images\photos\bupt.bmp')
contrast = cv2.imread(r'.\images\photos\buptstegoR.bmp')
res = psnr(original, contrast)
print(res)
