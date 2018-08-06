#!/usr/bin/env python
# encoding:utf-8

"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license. 
Please see LICENSE file in the project root for terms.
"""

"""
基于雅虎nsfw模型的色情图片识别,返回NSFW分数标识图片性质.

根据使用场景做了如下修改
1. 使用opencv的图片处理库替代了原声的StringIO
2. 支持通过调用Python函数获取结果(官方只提供了命令行方式)
3. 支持一次训练多次测试

author  :   @h-j-13
time    :   2018-8-6
ref     :   https://github.com/yahoo/open_nsfw
ref     :   https://blog.csdn.net/xingchenbingbuyu/article/details/52821497
ref     :   https://blog.csdn.net/xingchenbingbuyu/article/details/52821497

>>>train()  # 加载NSFW模型
>>>test('test.png') # 获取图片NSFW分数
0.0010637154337018728
"""

# 调整caffe的日志级别,只输出警告;关闭日志
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['GLOG_minloglevel'] = '3'

import sys
import glob
import time
import argparse
import numpy as np

# from PIL import Image
# from StringIO import StringIO  StringIO 2 opencv
import caffe
import cv2

# 训练数据全局变量
CAFFE_TRANSFORMER = None
NSFW_NET = None
HAS_TRAIN = False


def caffe_preprocess_and_compute(pimg, caffe_transformer=None, caffe_net=None,
                                 output_layers=None):
    """
    Run a Caffe network on an input image after preprocessing it to prepare
    it for Caffe.
    :param PIL.Image pimg:
        PIL image to be input into Caffe.
    :param caffe.Net caffe_net:
        A Caffe network with which to process pimg afrer preprocessing.
    :param list output_layers:
        A list of the names of the layers from caffe_net whose outputs are to
        to be returned.  If this is None, the default outputs for the network
        are returned.
    :return:
        Returns the requested outputs from the Caffe net.
    """
    if caffe_net is not None:
        # Grab the default output names if none were requested specifically.
        if output_layers is None:
            output_layers = caffe_net.outputs
        # repalce defined resize() by cv2.resize()
        img_data_rs = cv2.resize(pimg, (256, 256))
        cv2.imwrite('temp.jpg', img_data_rs)
        # img_data_rs = resize_image(pimg, sz=(256, 256))
        image = caffe.io.load_image('temp.jpg')

        H, W, _ = image.shape
        _, _, h, w = caffe_net.blobs['data'].data.shape
        h_off = max((H - h) / 2, 0)
        w_off = max((W - w) / 2, 0)
        crop = image[h_off:h_off + h, w_off:w_off + w, :]
        transformed_image = caffe_transformer.preprocess('data', crop)
        transformed_image.shape = (1,) + transformed_image.shape

        input_name = caffe_net.inputs[0]
        all_outputs = caffe_net.forward_all(blobs=output_layers,
                                            **{input_name: transformed_image})

        outputs = all_outputs[output_layers[0]][0].astype(float)
        return outputs
    else:
        return []


def train():
    """加载训练好的NFSW模型"""
    global HAS_TRAIN, CAFFE_TRANSFORMER, NSFW_NET
    if not HAS_TRAIN:
        pycaffe_dir = os.path.dirname(__file__)
        model_def = "nsfw_model/deploy.prototxt"
        pretrained_model = "nsfw_model/resnet_50_1by2_nsfw.caffemodel"

        # Pre-load caffe model.
        nsfw_net = caffe.Net(model_def,  # pylint: disable=invalid-name
                             pretrained_model, caffe.TEST)
        NSFW_NET = nsfw_net
        # Load transformer
        # Note that the parameters are hard-coded for best results
        caffe_transformer = caffe.io.Transformer({'data': nsfw_net.blobs['data'].data.shape})
        caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
        caffe_transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
        caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
        caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
        CAFFE_TRANSFORMER = caffe_transformer


def test(image_path):
    """测试图片,返回NSFW模型"""
    global CAFFE_TRANSFORMER, NSFW_NET
    start_time = time.time()
    image_data = cv2.imread(image_path)

    # Classify.
    scores = caffe_preprocess_and_compute(image_data, caffe_transformer=CAFFE_TRANSFORMER, caffe_net=NSFW_NET,
                                          output_layers=['prob'])

    # Scores is the array containing SFW / NSFW image probabilities
    # scores[1] indicates the NSFW probability
    end_time = time.time()
    print "NSFW\t:\t%.8f\tuse:\t%.6f\tsec." % (scores[1], end_time - start_time)
    return scores[1]


if __name__ == '__main__':
    s = time.time()
    train()
    print 'train nsfw modle used: %6.f' % (time.time() - s)
    for c in os.listdir('./data'):
        c_path = './data/' + c + '/'
        for image_name in os.listdir(c_path):
            print image_name,
            test(c_path + image_name)
