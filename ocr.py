#!/usr/bin/env python
# encoding:utf-8


"""
图片OCR(光学字符识别)功能
基于pyocr,pytesseract库开发

author  :   @h-j-13
time    :   2018-8-7
ref     :

>>>ocr('test.png',lang='chi_sim')
test 1234 qwer ':!0o
"""

import os
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

from pyocr import pyocr
from PIL import Image
import pytesseract


def ocr(image, lang='chi_sim'):
    """识别图像中的文字"""
    return pytesseract.image_to_string(Image.open(image), lang=lang)


# # 使用pyocr集成ocr环境工具
# tools = pyocr.get_available_tools()[:]
# tools[0].image_to_string(Image.open('ocr.py'),lang='chi_sim')
