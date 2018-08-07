#!/usr/bin/env python
# encoding:utf-8


"""
基于与opencv的人像(全身/人脸)识别
author  :   @h-j-13
time    :   2018-8-7
doc     :   https://github.com/opencv/opencv
ref     :   https://blog.csdn.net/kk185800961/article/details/79302193
ref     :   https://blog.csdn.net/haohuajie1988/article/details/79163318
"""

"""
DOC
备注：

1. detecMultiScale()函数

参数介绍：
参数1：image--待检测图片，一般为灰度图像加快检测速度；
参数2：objects--被检测物体的矩形框向量组；
参数3：scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
参数4：minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。
        如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
        如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
        这种设定值一般用在用户自定义对检测结果的组合程序上；
参数5：flags--要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为
        CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，
        因此这些区域通常不会是人脸所在区域；
参数6、7：minSize和maxSize用来限制得到的目标区域的范围。



2.人脸检测器
Opencv自带训练好的人脸检测模型，存储在sources/data/haarcascades文件夹和sources/data/lbpcascades文件夹下。其中几个.xml文件如下： 
人脸检测器（默认）：haarcascade_frontalface_default.xml 
人脸检测器（快速Harr）：haarcascade_frontalface_alt2.xml 
人脸检测器（侧视）：haarcascade_profileface.xml 
眼部检测器（左眼）：haarcascade_lefteye_2splits.xml 
眼部检测器（右眼）：haarcascade_righteye_2splits.xml 
嘴部检测器：haarcascade_mcs_mouth.xml 
鼻子检测器：haarcascade_mcs_nose.xml 
身体检测器：haarcascade_fullbody.xml 
人脸检测器（快速LBP）：lbpcascade_frontalface.xml
"""

import cv2

CLASSIFIER = None


def detect(image):
    """基于opencv内置的分类器识别人脸"""
    global CLASSIFIER
    # 加载分类器
    if not CLASSIFIER:
        classifier = cv2.CascadeClassifier(r'./model/haarcascades/haarcascade_frontalface_default.xml')
        CLASSIFIER = classifier
    # 读取本地用于识别的图片
    img = cv2.imread(image)
    # 多脸部识别，返回list。minSize 适当调整
    faces = CLASSIFIER.detectMultiScale(img, 1.1, 5)
    # 在图片img绘每个人的脸部方框
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 显示图片/切割,保存图片
    cv2.imshow("Image", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    detect('t.jpg')
