import skimage.feature
from tqdm import tqdm
from skimage import feature
import matplotlib.pyplot as plt
import numpy as np
import math

def load_data(data_path):
    ## 数据分为train、validate、test三种，
    ## 一般操作为：train_data 用于训练， validate_data 用于验证，通过调参，得到一个在validate_data上效果最好的模型。
    ## 在validate_data 上最好的模型，要去test_data上进行测试，得到测试的结果（指标），作为这个模型的真实水平。
    data = np.load(data_path)
    train_images = data["train_images"]
    train_labels = data["train_labels"]
    val_images = data["val_images"]
    val_labels = data["val_labels"]
    test_images = data["test_images"]
    test_labels = data["test_labels"]
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def processing(images):
    ## TODO 进行图像预处理（可以尝试多种方式，参考实验一）
    ## images 为一批图像，此处需要通过for循环对每个图像进行预处理操作，
    ## 可以参考hog_extraction函数的写法，也可以按自己的想法写，达到目标即可。
    img = np.zeros((len(images), len(images[0]), len(images[0][0])), dtype=np.int)
    for i in tqdm(range(len(images)), total=len(images)):
        for m in range(len(images[i])):
            for n in range(len(images[i][m])):
                    img[i][m][n] = (images[i][m][n][0]*299+images[i][m][n][1]*587+images[i][m][n][2]*114+500)/1000
    images = img
    images = images.astype(np.float32)
    for i in tqdm(range(len(images)), total=len(images)):
        images[i] = (images[i] - np.min(images[i])) / (np.max(images[i]) - np.min(images[i]))
    return images

def flatten_extraction(images):
    ## 直接将图像进行flatten作为特征输入到分类器。
    images = images.reshape(len(images), -1)
    return images

def hog_extraction(images, size):
    # 提取hog特征和对应的hog image图像 ，使用skimage中的feature.hog函数（请自行查阅相关内容）
    hogfeatures = []
    ## 使用for循环对传入的images进行hog特征提取。 tqdm 为进度条，可以方便查看进度。
    for i in tqdm(range(len(images)), total=len(images)):
        ## TODO 提取hog特征的函数，使用skimage中的feature.hog函数，
        ## 返回特征跟对应的hog图像（可以进行可视化展示），注意要想返回值为两个，则需要传入visualize=True
        hogfeature, image_hog = feature.hog(images[i], orientations=9, pixels_per_cell=(size, size),
                    cells_per_block=(2, 2), visualize=True)
        ## image_hog 为可以进行可视化的图像，检查一下提取的效果，真正训练时需要将此处注释，避免程序在此处卡住。
        # plt.subplot(1, 2, 1)
        # plt.imshow(images[i], cmap="gray")
        # plt.subplot(1, 2, 2)
        # plt.imshow(image_hog, cmap="gray")
        # plt.show()
        #将提取完的特征加入list中保存
        hogfeatures.append(hogfeature)
    ## 将提取的完的特征转为矩阵。
    hogfeatures = np.array(hogfeatures)
    return hogfeatures

def histogram_extraction(images):
    ## TODO 提取直方图特征的函数，计算各图像的直方图
    histfeature = []
    for i in tqdm(range(len(images)), total=len(images)):
        hist = np.histogram(images[i][:,:],256)
        histfeature.append(hist[0])
    histfeature = np.array(histfeature)
    return histfeature

class GLCM_extraction():
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
        self.images = images
        self.levels = levels
        self.dx = dx
        self.dy = dy
        self.kernel_size = kernel_size
        self.vmin = vmin
        self.vmax = vmax

    def cal_glcm(self):
        ## TODO 实现灰度共生矩阵计算
        row, col = self.images.shape
        # digitize input images to ensure that input images contain integers in [0, `levels`-1]
        ## TODO 数字化输入图像
        # 在此我将数字化步骤融合到GLCM计算步骤中
        self.images = self.images*256
        self.images = self.images.astype(np.int16)
        # calculate glcm
        ## TODO 补充完成 GLCM 计算
        srcdata = self.images.copy()
        glcm = np.zeros((self.levels,self.levels))
        if 255 > self.levels:
            for j in range(row):
                for i in range(col):
                    srcdata[j][i] = srcdata[j][i] * self.levels / 256
        if (self.dx < 0):
            for j in range(row - self.dy):
                for i in range(-self.dx,col + self.dx):
                    rows = srcdata[j][i] - 1
                    cols = srcdata[j + self.dy][i + self.dx] - 1
                    glcm[rows][cols] += 1.0
        else:
            for j in range(row - self.dy):
                for i in range(col - self.dx):
                    rows = srcdata[j][i]-1
                    cols = srcdata[j + self.dy][i + self.dx]-1
                    glcm[rows][cols] += 1.0
        a = glcm.sum(axis=(0, 1))
        for i in range(self.levels):
            for j in range(self.levels):
                glcm[i][j] /= a
        return glcm

    def cal_props(self, glcm):
        ## TODO 计算各项特征（对比度contrast、同质度homogeneity、相关性correlation、熵entropy）
        """ 可自行修改，仅为示例 """
        # properties = ['contrast', 'homogeneity', 'correlation', 'entropy']
        row, col = self.images.shape
        cont = 0
        homo = 0
        corr = 0
        entropy = 0
        mean = 0
        variance = 0

        for i in range(self.levels):
            for j in range(self.levels):
                mean += glcm[i, j] * i / (self.levels) ** 2
                variance += glcm[i, j] * (i - mean) ** 2

        for i in range(self.levels):
            for j in range(self.levels):
                ## TODO 根据公式计算各特征
                cont += glcm[i, j] * (i-j)**2
                homo += glcm[i, j] / (1.+(i-j)**2)
                corr += ((i - mean) * (j - mean) * (glcm[i, j]**2))/variance
                if glcm[i][j] > 0.0:
                    entropy += glcm[i][j] * math.log(glcm[i][j])
        return cont, homo, corr, entropy





