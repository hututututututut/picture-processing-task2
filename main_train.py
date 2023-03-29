import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
import math
from sklearn import svm
import joblib
from helper_function import processing, load_data, flatten_extraction, hog_extraction, histogram_extraction, GLCM_extraction
from test import glcm_extraction
import cv2

data_path = "./data/dermamnist.npz"
# data_path = "./data/pneumoniamnist.npz"

def proprecessing_data(train_images, val_images):
    ## TODO 对数据进行预处理，达到更好的分类结果；
    ## 具体预处理的代码写在helper_function.py中，在此处直接进行调用。
    train_images = processing(train_images)
    val_images = processing(val_images)
    return train_images, val_images

def extract_features(train_images, val_images):
    ## 1. 直接将图像进行flatten，作为特征输入到分类器分类。
    # train_images = flatten_extraction(train_images)
    # val_images = flatten_extraction(val_images)

    ## 2. 提取图像的hog特征，作为特征输入到分类器分类，TODO 需自行完善 hog_extraction 函数。
    # train_images = hog_extraction(train_images, size=7)
    # val_images = hog_extraction(val_images, size=7)

    ## 3. 提取图像的直方图特征. TODO 需自行完善 histogram_extraction 函数。
    # train_images = histogram_extraction(train_images)
    # val_images = histogram_extraction(val_images)
    ## 4. 提取图像的灰度共生矩阵特征. TODO 需自行完善 GLCM_extraction 函数。
    test = cv2.imread('D:\学习资料\大三\图像处理\Trial_2\\test.jpg')
    fast_glcm1 = glcm_extraction(images=test, levels=8, dx=0, dy=1)
    glcm1 = fast_glcm1.fast_glcm()
    mean, variance, cont, diss, homo, asm, ene, max_, corr, entropy = fast_glcm1.fast_glcm_props(glcm=glcm1)
    fast_glcm1.figuer(mean, variance, cont, diss, homo, asm, ene, max_, corr, entropy)

    a = np.zeros((len(train_images),16))
    b = np.zeros((len(val_images),16))
    angles = [0,math.pi/4,math.pi*2/4,3*math.pi/4]
    for angle in angles:
        for i in range(len(train_images)):
            fast_glcm = GLCM_extraction(images=train_images[i], levels=8, dx=round(math.cos(angle)), dy=round(math.sin(angle)))
            glcm = fast_glcm.cal_glcm()
            cont, homo, corr, entropy = fast_glcm.cal_props(glcm=glcm)
            a[i,int(16*angle/math.pi)], a[i,int(16*angle/math.pi+1)],a[i,int(16*angle/math.pi+2)],a[i,int(16*angle/math.pi+3)] = cont, homo, corr, entropy
        for i in range(len(val_images)):
            fast_glcm1 = GLCM_extraction(images=val_images[i], levels=8,  dx=round(math.cos(angle)), dy=round(math.sin(angle)))
            glcm = fast_glcm1.cal_glcm()
            cont, homo, corr, entropy = fast_glcm1.cal_props(glcm=glcm)
            b[i,int(16*angle/math.pi)], b[i,int(16*angle/math.pi+1)],b[i,int(16*angle/math.pi+2)],b[i,int(16*angle/math.pi+3)] = cont, homo, corr, entropy
    train_images = a
    val_images = b

    ## 5. 将提取的多种特征组合，进行分类. TODO 选做
    return train_images, val_images

if __name__ == '__main__':
    test = cv2.imread('D:\学习资料\大三\图像处理\Trial_2\\test.jpg')
    ## 加载数据
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data(data_path)
    """ 查看数据集构成，进行每个类别图像数据的预处理及特征提取 —— (4708,28,28) & (7007,28,28,3)"""
    ## 数据预处理
    train_images, val_images = proprecessing_data(train_images, val_images)
    ## 提取特征
    train_features, val_features = extract_features(train_images, val_images)

    ## 定义svm分类器
    model = svm.SVC()
    print(f"train_images shape is {train_images.shape}")
    print(f"val_images.shape is {val_images.shape}")

    print(f"train_features shape is {train_features.shape}")
    print(f"val_features.shape is {val_features.shape}")
    ## 训练
    model.fit(train_features, train_labels)
    ## 验证
    score = model.score(val_features, val_labels)
    print(f"score is {score}")
    ## 保存模型
    joblib.dump(model, "./svm_model.bin")


