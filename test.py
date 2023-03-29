import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math

class glcm_extraction():
    """
        完成灰度共生矩阵特征的提取（可以参考 skimage.feature.graycomatrix 中的源码，但需自行实现，不应挪用或直接调用函数）
        skimage.feature.graycomatrix 源码路径："D:\Anaconda\Lib\site-packages\skimage\feature\texture.py"
        以下仅为简单示例提醒，不是模板要求，可自行编写
    """

    def __init__(self, images, levels, dx, dy, kernel_size=5, vmin=0, vmax=255):
        """
        :param images: input image
        :param levels: int, number of grey-level of GLCM (e.g. 4, 8, 16 ...)，建议选取256的因数
        :param dx: pixel pair row pixel
        :param dy: pixel pair row pixel
        :param vmin: minimum pixel value of input image
        :param vmax: maximum pixel value of input image
        """
        self.images = images*756
        self.levels = levels
        self.dx = dx
        self.dy = dy
        self.kernel_size = kernel_size
        self.vmin = vmin
        self.vmax = vmax

    def fast_glcm(self):
        mi, ma = self.vmin, self.vmax
        ks = self.kernel_size
        h, w = self.images.shape
        # digitize input images to ensure that input images contain integers in [0, `levels`-1]
        ## TODO 数字化输入图像
        bins = np.linspace(mi, ma + 1, self.levels + 1)
        gl1 = np.digitize(self.images, bins) - 1
        gl2 = np.append(gl1[:, 1:], gl1[:, -1:], axis=1)
        ## TODO 补充完成 GLCM 计算
        glcm = np.zeros((self.levels, self.levels, h, w), dtype=np.uint8)
        for i in range(self.levels):
            for j in range(self.levels):
                mask = ((gl1 == i) & (gl2 == j))
                glcm[i, j, mask] = 1
        kernel = np.ones((ks, ks), dtype=np.uint8)
        # for i in range(self.levels):
        #     for j in range(self.levels):
        #         glcm[i, j] = cv2.filter2D(glcm[i, j], -1, kernel)
        glcm = glcm.astype(np.float32)
        return glcm

    def fast_glcm_props(self, glcm):
        row, col = self.images.shape
        cont = np.zeros((row, col), dtype=np.float32)
        homo = np.zeros((row, col), dtype=np.float32)
        corr = np.zeros((row, col), dtype=np.float32)
        entropy = np.zeros((row, col), dtype=np.float32)
        mean = np.zeros((row, col), dtype=np.float32)
        variance = np.zeros((row, col), dtype=np.float32)
        diss = np.zeros((row, col), dtype=np.float32)
        asm = np.zeros((row, col), dtype=np.float32)
        ## TODO 根据公式计算各特征
        for i in range(self.levels):
            for j in range(self.levels):
                mean += glcm[i, j] * i / (self.levels) ** 2
                variance += glcm[i, j] * (i - mean) ** 2

        for i in range(self.levels):
            for j in range(self.levels):
                cont += glcm[i, j] * (i - j) ** 2
                homo += glcm[i, j] / (1. + (i - j) ** 2)
                corr += ((i - mean) * (j - mean) * (glcm[i, j] ** 2)) / variance
                pnorm = glcm / np.sum(glcm, axis=(0, 1)) + 1. / self.kernel_size ** 2
                ent = np.sum(-pnorm * np.log(pnorm), axis=(0, 1))
                diss += glcm[i, j] * np.abs(i - j)
                asm += glcm[i, j] ** 2
        ene = np.sqrt(asm)
        max_ = np.max(glcm, axis=(0, 1))

        return mean, variance, cont, diss, homo, asm, ene, max_, corr, ent

    def figuer(self, mean, variance, cont, diss, homo, asm, ene, max_, corr, entropy):
        plt.figure(figsize=(10, 4.5))
        fs = 15
        plt.subplot(2, 5, 1)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(self.images, cmap=plt.cm.gray)
        plt.title('original', fontsize=fs)
        plt.subplot(2, 5, 2)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(mean)
        plt.title('mean', fontsize=fs)
        plt.subplot(2, 5, 3)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(variance)
        plt.title('variance', fontsize=fs)
        plt.subplot(2, 5, 4)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(cont)
        plt.title('contrast', fontsize=fs)
        plt.subplot(2, 5, 5)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(diss)
        plt.title('dissimilarity', fontsize=fs)
        plt.subplot(2, 5, 6)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(homo)
        plt.title('homogeneity', fontsize=fs)
        plt.subplot(2, 5, 7)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(asm)
        plt.title('ASM', fontsize=fs)
        plt.subplot(2, 5, 8)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(ene)
        plt.title('energy', fontsize=fs)
        plt.subplot(2, 5, 9)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(corr)
        plt.title('corr', fontsize=fs)
        plt.subplot(2, 5, 10)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(entropy)
        plt.title('entropy', fontsize=fs)
        plt.tight_layout(pad=0.5)
        plt.show()