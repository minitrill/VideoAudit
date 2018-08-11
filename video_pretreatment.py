#!/usr/bin/env python
# encoding:utf-8


"""
视频预处理模块
灰度化,缩放大小,视频截取

author  :   @h-j-13
time    :   2018-8-11
"""

from PIL import Image
from time import time

# 打开图片
i = Image.open('test.jpg')
s = time()
# 缩放
smaller_image = i.resize((16, 16))
smaller_image.save('1.jpg')

# 灰度化
grayscale_image = smaller_image.convert("L")
grayscale_image.save('2.jpg')

print time()-s