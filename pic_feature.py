#!/usr/bin/env python
# encoding:utf-8

"""
图像特征检测算法与相似图片搜索
基于SIFT,SURF,ORB特征结合opencv实现图片特征与相似图片搜索

author  :   @h-j-13
time    :   2018-8-8
ref     :   https://blog.csdn.net/sinat_26917383/article/details/63306206?locationNum=6&fps=1
ref     :   https://blog.csdn.net/samkieth/article/details/49615669
ref     :   https://blog.csdn.net/eds95/article/details/70146689

>>>image = cv2.imread('test.jpg')   # 通过OPENCV读取文件
>>>sfit(image)                      # 进行sfit特征变换
[[31.  1.  0. ... 92.  1.  0.]
 ...
[10. 30.  5. ...  0.  0.  0.]]
>>>orb(image)                       # 进行ORB特征变换
[[ 88 185  59 ...   0 230 186]
 ...
 [116 221  63 ... 123  37 122]]
>>>surf(image,'surf_test.jpg')      # 进行SURF特征变化,并保存提取特征后的图片
[[-3.3737912e-03 -1.9819981e-03  3.3737912e-03 ...  4.7051071e-04
   4.5059556e-03  1.3064195e-03]
   ....
>>>i1 = cv2.imread('test.jpg')
>>>i2 = cv2.imread('test2.jpg')
>>>print compare_pic(i1,i2,show=True)    # 进行特征比对
165
"""

import cv2
import numpy as np

# 特征变换全局对象
D = 200
SFIT = None
ORB = None
SURF = None

"""
SIFT
尺度不变特征转换(Scale-invariant feature transform或SIFT)
是一种电脑视觉的算法用来侦测与描述影像中的局部性特征，
它在空间尺度中寻找极值点，并提取出其位置、尺度、旋转不变量，
"""


def sfit(image, image_save_path=''):
    """
    图片进行sfit变换,生成特征点返回特征矩阵
    :param image: cv2.imread
    :param image_save_path: 如果指定了保存地址,则会将特征提取过的图片保存到制定的地址中
    :return:sfit变换后的特征向量(128维度xD)矩阵
    """
    global SFIT, D
    if SFIT is None:
        SFIT = cv2.xfeatures2d.SIFT_create(D)
    # 灰度化/sfit特征
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = SFIT.detectAndCompute(gray, None)
    # 保存特征图片
    if image_save_path:
        image = cv2.drawKeypoints(gray, kp, image)
        cv2.imwrite(image_save_path, image)
    return des


"""
一种新的具有局部不变性的特征 —— ORB特征，
从它的名字中可以看出它是对FAST特征点与BREIF特征描述子的一种结合与改进
"""


def orb(image, image_save_path=''):
    """
    图片进行orb变换,生成特征点返回特征矩阵
    :param image: cv2.imread
    :param image_save_path: 如果指定了保存地址,则会将特征提取过的图片保存到制定的地址中
    :return:orb变换后的特征向量(32维度xD)矩阵
    """
    global ORB, D
    if ORB is None:
        ORB = cv2.ORB_create(D)
    # 灰度化/sfit特征
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = ORB.detectAndCompute(gray, None)
    # 保存特征图片
    if image_save_path:
        image = cv2.drawKeypoints(gray, kp, image)
        cv2.imwrite(image_save_path, image)
    return des


"""
SURF
采用快速Hessian算法检测关键点+提取特征。 
Surf在速度上比sift要快许多，这主要得益于它的积分图技术，
已经Hessian矩阵的利用减少了降采样过程，另外它得到的特征向量维数也比较少，
有利于更快的进行特征点匹配。
"""


def surf(image, image_save_path=''):
    """
    图片进行surf变换,生成特征点返回特征矩阵
    :param image: cv2.imread
    :param image_save_path: 如果指定了保存地址,则会将特征提取过的图片保存到制定的地址中
    :return:surf变换后的特征向量(64维度xn)矩阵
    """
    global SURF
    if SURF is None:
        SURF = cv2.xfeatures2d.SURF_create()
    # 灰度化/sfit特征
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = SURF.detectAndCompute(gray, None)
    # 保存特征图片
    if image_save_path:
        image = cv2.drawKeypoints(gray, kp, image)
        cv2.imwrite(image_save_path, image)
    return des


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


def compare_pic(image1, image2, show=False):
    """
    比较两张图片的特征
    :param image1: cv2.imread
    :param image2: cv2.imread
    :param draw: 是否要显示匹配的结果
    :return:匹配的特征点的个数
    """
    global SFIT, D
    if SFIT is None:
        SFIT = cv2.xfeatures2d.SIFT_create(D)
    # sift
    kp1, des1 = SFIT.detectAndCompute(image1, None)
    kp2, des2 = SFIT.detectAndCompute(image2, None)

    # BFmatcher with default parms
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # 匹配点
    goodMatch = []
    for m, n in matches:
        if m.distance < 0.50 * n.distance:
            goodMatch.append(m)

    # show
    if show:
        drawMatchesKnn_cv2(image1, kp1, image2, kp2, goodMatch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return len(goodMatch)


if __name__ == '__main__':
    import os
    import time

    class_name = 'data/normal/'
    image_list = []
    for file_name in os.listdir(class_name):
        image_list.append(cv2.imread(class_name + file_name))

    for i in xrange(len(image_list) - 1):
        start = time.time()
        print compare_pic(image_list[i], image_list[i + 1]),
        print time.time() - start

    # image = cv2.imread('test.jpg')
    # print sfit(image)
    # image = cv2.imread('test.jpg')
    # print orb(image)
    # image = cv2.imread('test.jpg')
    # print surf(image)
