#!/usr/bin/env python
# encoding:utf-8


"""
图片Dhash实现
dhash是一种感知哈希算法,可以用于相似图片检索及生成图片特征

感知hash算法分类
aHash：平均值哈希。速度比较快，但是常常不太精确。
pHash：感知哈希。精确度比较高，但是速度方面较差一些。
dHash：差异值哈希。Amazing！精确度较高，且速度也非常快。因此我就选择了dHash作为我图片判重的算法。

aHash:
均值哈希算法主要是利用图片的低频信息，其工作过程如下：
（1）缩小尺寸：去除高频和细节的最快方法是缩小图片，将图片缩小到8x8的尺寸，总共64个像素。不要保持纵横比，
只需将其变成8*8的正方形。这样就可以比较任意大小的图片，摒弃不同尺寸、比例带来的图片差异。
（2）简化色彩：将8*8的小图片转换成灰度图像。
（3）计算平均值：计算所有64个像素的灰度平均值。
（4）比较像素的灰度：将每个像素的灰度，与平均值进行比较。
大于或等于平均值，记为1；小于平均值，记为0。
（5）计算hash值：将上一步的比较结果，组合在一起，就构成了一个64位的整数，
这就是这张图片的指纹。组合的次序并不重要，只要保证所有图片都采用同样次序就行了。

pHash：
（1）缩小尺寸：pHash以小图片开始，但图片大于8*8，32*32是最好的。
（2）简化色彩：将图片转化成灰度图像，进一步简化计算量。
（3）计算DCT：计算图片的DCT变换，得到32*32的DCT系数矩阵。
（4）缩小DCT：虽然DCT的结果是32*32大小的矩阵，但我们只要保留左上角的8*8的矩阵，这部分呈现了图片中的最低频率。
（5）计算平均值：如同均值哈希一样，计算DCT的均值。
（6）计算hash值：这是最主要的一步，根据8*8的DCT矩阵，设置0或1的64位的hash值，大于等于DCT均值的设为”1”，
小于DCT均值的设为“0”。组合在一起，就构成了一个64位的整数，这就是这张图片的指纹。

dHash:
（1）缩小尺寸：dHash以小图片开始
（2）简化色彩：将图片转化成灰度图像，进一步简化计算量。
（3）差异计算：差异值是通过计算每行相邻像素的强度对比得出的。我们的图片为9*8的分辨率，那么就有8行，每行9个像素。
差异值是每行分别计算的，也就是第二行的第一个像素不会与第一行的任何像素比较。每一行有9个像素，那么就会产生8个差异值
（4）计算hash值:我们将差异值数组中每一个值看做一个bit，每8个bit组成为一个16进制值，
将16进制值连接起来转换为字符串，就得出了最后的dHash值。


author  :   @h-j-13
time    :   2018-8-7
ref     :   https://blog.csdn.net/haluoluo211/article/details/52769325
ref     :   http://blog.sina.com.cn/s/blog_56fd58ab0102xpqf.html
ref     :   https://www.cnblogs.com/faith0217/articles/4088386.html

>>>from PIL import Image
>>>i1 = Image.open('1.jpg')
>>>i2 = Image.open('2.jpg')
>>>DHash.calculate_hash(i1)
0038b694d5ba5538
>>>DHash.hamming_distance(i1,i2)
28
"""

from PIL import Image


class DHash(object):
    @staticmethod
    def calculate_hash(image):
        """
        计算图片的dHash值
        :param image: PIL.Image
        :return: dHash值,string类型
        """
        difference = DHash.__difference(image)
        # 转化为16进制(每个差值为一个bit,每8bit转为一个16进制)
        decimal_value = 0
        hash_string = ""
        for index, value in enumerate(difference):
            if value:  # value为0, 不用计算, 程序优化
                decimal_value += value * (2 ** (index % 8))
            if index % 8 == 7:  # 每8位的结束
                hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))  # 不足2位以0填充。0xf=>0x0f
                decimal_value = 0
        return hash_string

    @staticmethod
    def hamming_distance(first, second):
        """
        计算两张图片的汉明距离(基于dHash算法)
        :param first: Image或者dHash值(str)
        :param second: Image或者dHash值(str)
        :return: hamming distance. 值越大,说明两张图片差别越大,反之,则说明越相似
        """
        # A. dHash值计算汉明距离
        if isinstance(first, str):
            return DHash.__hamming_distance_with_hash(first, second)

        # B. image计算汉明距离
        hamming_distance = 0
        image1_difference = DHash.__difference(first)
        image2_difference = DHash.__difference(second)
        for index, img1_pix in enumerate(image1_difference):
            img2_pix = image2_difference[index]
            if img1_pix != img2_pix:
                hamming_distance += 1
        return hamming_distance

    @staticmethod
    def __difference(image, ):
        """
        *Private method*
        计算image的像素差值
        :param image: PIL.Image
        :return: 差值数组。0、1组成
        """
        resize_width = 9
        resize_height = 8
        # 1. resize to (9,8)
        smaller_image = image.resize((resize_width, resize_height))
        # 2. 灰度化 Grayscale
        grayscale_image = smaller_image.convert("L")
        # 3. 比较相邻像素
        pixels = list(grayscale_image.getdata())
        difference = []
        for row in range(resize_height):
            row_start_index = row * resize_width
            for col in range(resize_width - 1):
                left_pixel_index = row_start_index + col
                difference.append(pixels[left_pixel_index] > pixels[left_pixel_index + 1])
        return difference

    @staticmethod
    def __hamming_distance_with_hash(dhash1, dhash2):
        """
        *Private method*
        根据dHash值计算hamming distance
        :param dhash1: str
        :param dhash2: str
        :return: 汉明距离(int)
        """
        difference = (int(dhash1, 16)) ^ (int(dhash2, 16))
        return bin(difference).count("1")


if __name__ == '__main__':
    import os
    import time

    class_name = 'data/video/'
    image_list = []
    for file_name in os.listdir(class_name):
        image_list.append(Image.open(class_name + file_name))

    for i in xrange(len(image_list) - 1):
        start = time.time()
        print DHash.hamming_distance(image_list[i], image_list[i + 1]),
        print time.time() - start
