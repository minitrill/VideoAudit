#!/usr/bin/env python
# encoding:utf-8


"""
图片md5值功能
将一个任意类型的图片映射成MD5值,用于图片指纹

author  :   @h-j-13
time    :   2018-8-7

>>>image_md5('test.jpg')                    # 生成图片MD5值
1ae5226b5a371fecdc7f050beb828fb4

>>>M = MaliciousImage()                     # 初始化恶意图片MD5数据类
>>>M.add_image_md5(image_md5('1.jpg'))      # 添加恶意MD5值
>>>M.add_image_md5(image_md5('1.bmp'),image_type='pron')      # 添加恶意MD5值,并指明类型
>>>print M.has_key(image_md5('2.bmp'))      # 检测该图片MD5是否被收录
False
>>>print M.has_key(image_md5('1.bmp'))      # 检测该图片MD5是否被收录
True
>>>M.save2disk()                            # 数据持久化到本地磁盘
"""

import hashlib
import cPickle as pickle


class MaliciousImage(object):
    """恶意图片检测"""

    def __init__(self):
        """构造函数"""
        self.malicious_image_dict = {}
        self.read_data_from_dist()

    def __len__(self):
        """支持len获取长度"""
        return len(self.malicious_image_dict)

    def add_image_md5(self, image_md5, image_type='malicious'):
        """添加恶意图片"""
        self.malicious_image_dict[image_md5] = image_type

    def has_key(self, image_md5):
        """检测是否含有某个md5值"""
        return self.malicious_image_dict.has_key(image_md5)

    def save2disk(self, file_path='./data/malicious_image.dat'):
        """保存当前恶意图片MD5数据到磁盘中"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.malicious_image_dict, f)

    def read_data_from_dist(self, file_path='./data/malicious_image.dat'):
        """从磁盘中读取恶意图片MD5数据"""
        with open(file_path, 'rb') as f:
            self.malicious_image_dict = pickle.load(f)


def image_md5(image):
    """获取图片MD5值"""
    with open(image, "rb") as i:
        fmd5 = hashlib.md5(i.read())
        return fmd5.hexdigest()
