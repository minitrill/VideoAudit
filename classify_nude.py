#!/usr/bin/env python
# encoding:utf-8


"""
基于nude库来鉴别色情图片 (基于裸露皮肤程度-效果较差且无法量化)
author    :   @h-j-13
time      :   2018-7-21
ref       :   https://blog.csdn.net/qq_42022255/article/details/80349112
"""

import time

import nude


class ClassifyNude(object):
    """基于nude的图片分类器 图片 -> (裸露,非裸露)"""

    def test(self, pic_path):
        """
        监测图片是否为暴露图片
        :param pic_path:
        :return:Ture/False
        """
        start = time.time()
        result = nude.is_nude(pic_path)
        print pic_path + " 该图片"
        return result


print nude.is_nude('./data/pron/1.jpg')
print nude.is_nude('./data/pron/2.jpg')
print nude.is_nude('./data/pron/3.jpg')
