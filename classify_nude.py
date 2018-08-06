#!/usr/bin/env python
# encoding:utf-8


"""
基于nude库来鉴别色情图片 (基于裸露皮肤程度-效果较差且无法量化)
author    :   @h-j-13
time      :   2018-7-21
ref       :   https://blog.csdn.net/qq_42022255/article/details/80349112

>>>test('1.png')
True
"""

import time

import nude


def test(pic_path):
    """
    基于nude的图片分类器 图片 -> (裸露,非裸露)
    :param pic_path:
    :return:Ture/False
    """
    start = time.time()
    result = nude.is_nude(pic_path)
    end = time.time()
    print '\tresult\t:\t' + str(result) + "\tused\t:\t%.6f" % (end - start)
    return result


if __name__ == '__main__':
    import os

    for c in os.listdir('./data'):
        c_path = './data/' + c + '/'
        for image_name in os.listdir(c_path):
            print image_name,
            test(c_path + image_name)
