视频审核模块
========
minitrill 视频审核模块

## 模块思路
思路
![13.png](https://upload-images.jianshu.io/upload_images/5617720-75978e7019fd77b9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

架构
![](https://upload-images.jianshu.io/upload_images/5617720-90888a1421c5fb12.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 审核范围

- 用户上传的视频
- 用户的头像,上传的图片(*?*)

### 分析
与言论相比
1. 制作一个视频的成本更大,很少有人会因为非利益驱动而制作大量非法视频
2. 视频的传播速度快,覆盖面积大       
视频处理主要以**删除,封禁为主;锁定为辅**,以求尽快减少平台上的恶意视频

恶意视频性质分类
1. 色情视频
2. 低俗视频
3. 反动视频
4. 其他恶意视频

创作类型
1. 反动视频,暴恐视频 - 多为**原创**
2. 色情视频,低俗视频 - 多为转载,截取和二次加工

### 核心思路
**将难以处理的视频,难以衡量的色情程度进行转化,转化为其他易于衡量大小的数据**       

视频 -> 图片(抽取封面与关键帧)        
图片 -> 指纹/向量(用于相似度检索及审核去重)           
图片恶意 -> 恶意值
- 色情程度      
    * 肌肤裸露比例      
    * 人物造型　       
- 反动程度
    * 关键图像出现次数
    * 视频评论关键词

### 核心问题
1. 如何检测出恶意视频/图片(主要指色情图片)
2. 如何防止某特定主题恶意视频集中上传/过审     
3. **如何生成一个表明视频特征的指纹?**
4. 如何检测反动及暴恐视频(不通过图片识别)


## 功能
### 1. 色情图片识别

### 基于皮肤识别算法的色情图片识别 
本方法是一个朴素方法,效果较差,但可以用来打开这方面的思路.进行色情图片识别最好的方法还是深度学习

**算法描述**        
程序的关键步骤如下           
1. 遍历每个像素，检测像素颜色是否为肤色
2. 将相邻的肤色像素归为一个皮肤区域，得到若干个皮肤区域
3. 剔除像素数量极少的皮肤区域

![before.png](https://upload-images.jianshu.io/upload_images/5617720-112fc0097c99f8f1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![after.png](https://upload-images.jianshu.io/upload_images/5617720-d15f9dc17bebd415.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

皮肤检测算法
1. RGB 颜色模式
```
r > 95 and g > 40 and g < 100 and b > 20 and 
max([r, g, b]) - min([r, g, b]) > 15 and 
abs(r - g) > 15 and r > g and r > b
```
```
nr = r / (r + g + b), 
ng = g / (r + g + b), 
nb = b / (r +g + b), 

nr / ng > 1.185 and r * b / (r + g + b) ** 2 > 0.107 and 
r * g / (r + g + b) ** 2 > 0.112
```

HSV 颜色模式
```
h > 0 and h < 35 and s > 0.23 and s < 0.68
```

YCbCr 颜色模式
```
97.5 <= cb <= 142.5 and 134 <= cr <= 176
```

我们定义非色情图片的判定规则如下（满足任意一个判定为真）        
1. 皮肤区域的个数小于 3 个
2. 皮肤区域的像素与图像所有像素的比值小于 15%
3. 最大皮肤区域小于总皮肤面积的 45%
4. 皮肤区域数量超过60个

**算法实现**        
程序入口见`pic_classify_skin.py`,这里基于PIL来处理图像
```python
>>>n = Nude('test.jpg')     # 创建对象
>>>n.parse()                # 识别图片
>>>print n.result           # 识别结果 T/F
>>>n.showSkinRegions()      # 保存皮肤图片结果
```

**缺点**
1. 时间长(处理图片越需要3~5s)
2. 皮肤识别算法过于简单,无法识别皮肤过深,过浅及黑人
3. 色情图片判定的**规则过于简单**(如果这里能结合判断各个皮肤模块间的造型与关系效果则会变得更好)
4. 难以区分鉴别色情图片与低俗图片

例如,由于皮肤识别算法局限所造成的误判
![jk.png](https://upload-images.jianshu.io/upload_images/5617720-9def0f544d0e24ff.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)![jk2.png](https://upload-images.jianshu.io/upload_images/5617720-fd1c58c418a42211.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

经过了这个算法的实践,我们可以得到一些结论
1. 判断皮肤是色情图片识别一个比较重要的方法
2. 色情图片中9成以上是**过度暴露的女性**,这一点应该是接下来建模的重点考虑的因素

### 基于nude(裸露程度)的色情图片识别
[nudepy](https://pypi.org/project/nudepy/) 这个库基本上可以视为上述方法的威力加强版
库内通过c语言实现了一个皮肤分类器,并基于较复杂的裸露程度来判别图片是否是色情图片

**说明**      
程序入口见`pic_classify_nude.py`,这里主要是对于nude库的封装
```python
>>>from pic_classify_nude import test
>>>
>>>test('1.png')    # 判断色情图片T/F
True
```

**性能测试**        
这里收集了约300张图片,共分为色情(漏点),低俗(不漏点),普通图片三类,每类100张左右

| 类别 | 鉴别率/误判率(%) | 平均处理时间(s) |
| :------: | :------ | :------ |
| 普通 | 10.12 | 6.078 |
| 低俗 | 68.3 | 5.2754 |
| 色情 | 81.63 | 1.9179 |

识别率尚可,主要是处理时间过长,且每张图片只能给出是/不是的判断,无法量化

### 基于yahoo的NSFW模型的色情图片识别
yahoo在2016年开源了他们的色情图片识别模型 [Open nsfw model](https://github.com/yahoo/open_nsfw)
这里面有详细的介绍,这里就不赘述了.主要是基于caffe平台结合深度学习与计算机视觉所构建的深度神经网络.   
只针对与真人的色情图片,对于漫画,草图或者其他拟人neta色情识别性能较差

**说明**      
这里对nsfw进行了如下修改
1. 开发了nsfw模型的python内部调用接口   
nsfw官方只提供了命令行方式调用,
```shell
python ./classify_nsfw.py \
--model_def nsfw_model/deploy.prototxt \
--pretrained_model nsfw_model/resnet_50_1by2_nsfw.caffemodel \
INPUT_IMAGE_PATH 
```
通过修改`classify_nsfw.py`,实现了通过python接口调用的功能,并可以一次训练多次测试
(之前的逻辑是每次训练都要重新加载数据)

2. 重写了nsfw内的图片处理resize()函数  
原生nsfw调用的是`StringIO`的图片处理库,在windows环境下出现了以下Exception
`IOError: cannot identity image file<StringIO.StringIO instance at ...>`
这里查阅资料后,改用了OPENCV库重写了nsfw内处理图片的逻辑. 提高了程序健壮性但要求要提前安装opencv

**使用说明**            
程序入口见`pic_classify_nsfw.py`,调用方式如下
```python
>>>train()          # 加载NSFW模型(只需加载一次模型)
>>>test('test.png') # 获取图片NSFW分数
0.0010637154337018728
```

**性能测试**        
测试数据与nudepy数据相同

| 类别 | 鉴别率/误判率(%) | 平均处理时间(s) | 最大NSFW值 | 最小NSFW值 | 平均NSFW值 | NSFW值标准差 |
| :------: | :------ | :------ | :------ | :------ | :------ | :------ |
| 普通 | 1.23 | 0.12886 | 0.12886 | 0.50189018 | 0.015123606 | 0.003818056 | 
| 低俗 | 88.57 | 0.13634 | 0.97121823 | 0.06100044 | 0.536057051 | 0.05780393 | 
| 色情 | 96.20 | 0.12024 | 0.99999809 | 0.07825673 | 0.907023821 | 0.040200535 |


**总结**      
得到结果后手动查看发现,nsfw除了上面提到的不适应的场景之外,对于人物占图片整体面积较小的色情图片识别效果也比较差.
这里可以结合人物识别之后再进nsfw进行判断效果较好.

#### 基于opencv的人脸(人像)识别
基于上面所提到的局限性,现在希望能够从图片中把人体提取出来已增加人体区域所占图片大小的比例来提高nsfw模型的识别效果
这里基于opencv所提供的训练模型开发了识别图片中人类身体特征的代码

**程序说明**        
程序入口为`body_recongnition.py`,功能函数为`detect()`     
部分调参说明及核心函数详见代码中的注释及Doc

**功能测试**        
以人脸识别为例 

![g2.png](https://upload-images.jianshu.io/upload_images/5617720-aac7ba4fbb4e0a34.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)![g.png](https://upload-images.jianshu.io/upload_images/5617720-639d58cb6e1e5967.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

发现对于以人为主体的图片识别效果较好,但是对于人物占比较小的图片的识别能力一般,经常发生误判,
这里可能还需要一个调参的过程才能发挥较好的效果,由于时间关系就暂时不在推进这个方向了.


### 2. 图片指纹(MD5/Dhash)

### 基于MD5的图片指纹
这里是用md5来对图片的二进制数据来进行md5得到图片指纹
对于输入的图片先进行`灰度化`,紧接着`尺寸缩放`,最后根据图片的二进制数据结合`MD5`生成图片指纹 
这样的处理逻辑可以使得一个MD5覆盖更多的同类图片(主要是针对不同分辨率,清晰图的相同图片)


**程序说明**
程序入口`pic_md5.py`,函数入口`image_md5()`,只需要输入图片路径即可得到对应的md5值

例如      
mario.jpg       
![](https://upload-images.jianshu.io/upload_images/5617720-9ebdfae0e85495a0.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```python
>>>from pic_md5 import image_md5
>>>image_md5('mario.jpg')
>>>1ae5226b5a371fecdc7f050beb828fb4
```

推荐使用方法
```python
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
```

**优缺点**
```diff
+ 实现简单,可跨平台
+ 相同图片即使更改后缀或简单的格式转换也不会改变MD5值
+ 基于Python自建的dict实现,查询为O(1),每次查询仅为 10^-6s
- 图片通过第三方工具进行格式转换的时候会变更MD5值(jpg->bmp)
- 相同内容图片但是分辨率不同会生成不同的MD5值
```

### 基于dHash的图像指纹
针对上面提到的一个图片只要进行微小的改变就会生成新的MD5值,希望能够扩大一个图片指纹所能覆盖的恶意图片范围
这里采用了[感知哈希算法](https://www.cnblogs.com/faith0217/articles/4088386.html)中表现最好的[dHash](https://www.jianshu.com/p/193f0089b7a2)来实现图片指纹生成及相似图片检索的功能

**算法原理**
感知hash算法分类  
- aHash：平均值哈希。速度比较快，但是常常不太精确。
- pHash：感知哈希。精确度比较高，但是速度方面较差一些。
- dHash：差异值哈希。Amazing！精确度较高，且速度也非常快。因此我就选择了dHash作为我图片判重的算法。

1. aHash    
均值哈希算法主要是利用图片的低频信息，其工作过程如下： 
（1）缩小尺寸：去除高频和细节的最快方法是缩小图片，将图片缩小到8x8的尺寸，总共64个像素。不要保持纵横比， 
只需将其变成8\*8的正方形。这样就可以比较任意大小的图片，摒弃不同尺寸、比例带来的图片差异。  
（2）简化色彩：将8\*8的小图片转换成灰度图像。    
（3）计算平均值：计算所有64个像素的灰度平均值。       
（4）比较像素的灰度：将每个像素的灰度，与平均值进行比较。       
大于或等于平均值，记为1；小于平均值，记为0。     
（5）计算hash值：将上一步的比较结果，组合在一起，就构成了一个64位的整数，        
这就是这张图片的指纹。组合的次序并不重要，只要保证所有图片都采用同样次序就行了。        

2. pHash        
（1）缩小尺寸：pHash以小图片开始，但图片大于8\*8，32\*32是最好的。     
（2）简化色彩：将图片转化成灰度图像，进一步简化计算量。        
（3）计算DCT：计算图片的DCT变换，得到32\*32的DCT系数矩阵。        
（4）缩小DCT：虽然DCT的结果是32\*32大小的矩阵，但我们只要保留左上角的8\*8的矩阵，这部分呈现了图片中的最低频率。      
（5）计算平均值：如同均值哈希一样，计算DCT的均值。     
（6）计算hash值：这是最主要的一步，根据8\*8的DCT矩阵，设置0或1的64位的hash值，大于等于DCT均值的设为”1”，        
小于DCT均值的设为“0”。组合在一起，就构成了一个64位的整数，这就是这张图片的指纹。                

3. dHash
（1）缩小尺寸：dHash以小图片开始     
（2）简化色彩：将图片转化成灰度图像，进一步简化计算量。            
（3）差异计算：差异值是通过计算每行相邻像素的强度对比得出的。我们的图片为9\*8的分辨率，那么就有8行，每行9个像素。     
差异值是每行分别计算的，也就是第二行的第一个像素不会与第一行的任何像素比较。每一行有9个像素，那么就会产生8个差异值      
（4）计算hash值:我们将差异值数组中每一个值看做一个bit，每8个bit组成为一个16进制值，       
将16进制值连接起来转换为字符串，就得出了最后的dHash值。     

**举例说明dHash**

1. 原图片     
![1.jpg](https://upload-images.jianshu.io/upload_images/5617720-ba2007532dc9b2fb.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
        
2. 缩小尺寸        
![2.png](https://upload-images.jianshu.io/upload_images/5617720-70e13187b18212cc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
        
3. 灰度化         
![3.png](https://upload-images.jianshu.io/upload_images/5617720-4081e49c45f0e22d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

4. 差异计算
```
01010101010101010110101101...
....
```

5. 哈希值          
`1ae5226b5a371fecdc7f050beb828fb4`


**说明**
程序入口见`pic_Dhash.py`中,`DHash`所提供的`calculate_hash()`,`hamming_distance()`方法
使用之前需要借助python的PIL库来读取图片,使用方式如下

```python
>>>from PIL import Image
>>>i1 = Image.open('1.jpg')
>>>i2 = Image.open('2.jpg')
>>>DHash.calculate_hash(i1)
0038b694d5ba5538
>>>DHash.hamming_distance(i1,i2)
28
```

**性能测试**
这里主要测试Dhash对于计算速度及相似图片的dhash汉明距离
这里选取了约300张毫无关联的图片以及一个短视频每隔0.33s进行截图所生产的图片集
将这两个图片集分别命名为`非关联`与`关联`  
此外这个的时间是将两个图片**生成dHash值并进行汉明距离比较的时间之和**

| 类别 | 平均汉明距离 | 平均处理时间(s) | 最大汉明距离 | 最小汉明距离 |
| :------: | :------ | :------ | :------ | :------ |
| 非关联 | 31.06 | 0.0292 | 42 | 10
| 关联 | 10 | 0.0047 | 19 |2| 

可以看出对于关联与非关联图片这里还是有一定区分度的,下面直观的举例来说明汉明距离与图片的关系

1. 汉明距离<5(可以视为是相似图片)

* 第一对    
![b-001.jpg](https://upload-images.jianshu.io/upload_images/5617720-35709ff9392de9e4.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)![b-002.jpg](https://upload-images.jianshu.io/upload_images/5617720-ee5f1276ee6d90fb.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* 第二对       
![b-004.jpg](https://upload-images.jianshu.io/upload_images/5617720-31f2be42b20e8252.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)![b-005.jpg](https://upload-images.jianshu.io/upload_images/5617720-9397847a6a02369d.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


2. 汉明距离=17

![b-017.jpg](https://upload-images.jianshu.io/upload_images/5617720-e659d1a751415843.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)![b-016.jpg](https://upload-images.jianshu.io/upload_images/5617720-ee309a7c28c127ba.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### 3. 图片相似检索
构建一个恶意视频库,通过图片指纹与相似度技术结合hash索引进行快速检索使这里的核心目标
其实上文提到的dHash结合一定的优化(例如hash值分段然后结合树或者hash表)已经能承担相似图片检索的功能
下面是一些比较流行的图像特征化思路及结合相思图片检索的内容

参考网上大牛的文章,图像检索主要分为以下几个方向,上面讲述的感知哈希就是流行的方向之一

>1、方向一：描述符。先抽取图像局部特征符，根据一些匹配算法进行匹对，较多适合在图像比对，人脸比对等领域；       
>2、方向二：描述符的特征。先抽取图像局部特征符，描述符堆砌然后降维下来，成为拥有全局信息的图像特征向量，适合形成图指纹；   
>3、方向三：颜色特征。使用直方图，后计算两两之间的误差，较多使其向量化，比较适合多量图像。     
>4、方向四：哈希化。适合大规模搜索，哈希化之后去重，或进行搜索。   
>5、方向五：triplet loss。深度学习，从图片相似度学习图片的表示   

> 摘自 [实例检索︱图像的实例搜索（文献、方法描述、商业案例](https://blog.csdn.net/sinat_26917383/article/details/63306206?locationNum=6&fps=1)

### 基于图片特征向量的图片相似度检索

主要基于以下三种算法来提取图片特征       
- SIFT特征（尺度不变特征变换，scale-Invariant feature transform）
- SURF特征
- ORB特征
详细的介绍可以参考`pic_feature.py`中的DOC字符串

**程序说明**        
入口见`pic_feature.py`,推荐使用方法            
```python
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
>>>print compare_pic(i1,i2,show=True)    # 进行特征比对,返回匹配的特征数量
165
```

**效果展示**        
1. 原图片      
![1.jpg](https://upload-images.jianshu.io/upload_images/5617720-86b1001571fc0d18.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


2. sfit特征效果     
![2.jpg](https://upload-images.jianshu.io/upload_images/5617720-638239a286055d10.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


3. orb特征效果      
![3.jpg](https://upload-images.jianshu.io/upload_images/5617720-5d3ac2e4eae3c8ae.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


4. surf特征效果     
![4.jpg](https://upload-images.jianshu.io/upload_images/5617720-f520c6643948bdc9.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**性能测试**        
测试集同上,分为非关联,关联图片集

| 类别 | 平均匹配特征个数 | 平均处理时间(s) | 最大匹配特征个数 | 最小匹配特征个数 |
| :------: | :------ | :------ | :------ | :------ |
| 非关联 | 0.2658 | 0.6310 | 26 | 0 |
| 关联 | 50.4 | 0.1665 | 166 | 17 | 
    
**总结**        
通过`匹配特征数>(50~100)`基本可以确定是一张相似图片,不过这里的           
处理时间很大程度上受到了图片质量的影响,甚至**低于nsfw模型判定的时间**.        
如果结合之前学习的faiss的库或许能够提升对于大规模图片匹配的速度.
但是这个功能的适用场景是否符合我们的要求这里还需确定 *?*     
        
**图片相似测试**      
1. 同一图片进行处理(旋转,缩放)      
![3.png](https://upload-images.jianshu.io/upload_images/5617720-6dee6decc0de8307.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2. 不同图片     
![3222.png](https://upload-images.jianshu.io/upload_images/5617720-1fd52044d29993d4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

3. 完全相同的图片      
![4444.png](https://upload-images.jianshu.io/upload_images/5617720-6ace802dcc469d27.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 4. 图片OCR
这里主要基于[Tesseract](https://github.com/tesseract-ocr)开发了图片OCR的功能
使用之前需要预先[安装Tesseract环境](https://github.com/UB-Mannheim/tesseract/wiki)

> Tesseract 是一个 OCR 库,目前由 Google 赞助(Google 也是一家以 OCR 和机器学习技术闻名于世的公司)。       
Tesseract 是目前公认最优秀、最精确的开源 OCR 系统。 除了极高的精确度,Tesseract 也具有很高的灵活性。     
它可以通过训练识别出任何字体，也可以识别出任何 Unicode 字符。

**程序说明**        
入口见`ocr.py`
```python
>>>ocr('test.png',lang='chi_sim')   # 制定图片路径,默认解析简体中文
test 1234 qwer ':!0o
```

**测试**      
1. 多文字测试  
      
原图          
![1.png](https://upload-images.jianshu.io/upload_images/5617720-4e2ea6f012923af9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

识别结果        
```text
We choose to go to the moon in this decade and do the other
things, not because they are easy, but because they are hard，
because that goal will serve to organize and measure the best of
our energies and skills, because that challenge is one that we are
willing to accept, one we are unwiling to postpone, and one which
we intend to win, and the others, too.

我们决定在这十年间登上月球并实现更多梦想，并非它们轻而
易举，而正是因为它们困难重重。因为这个目标将促进我们实现最
佳的组织并测试我们顶尖的技术和力量， 因为这个挑战我们乐于接
受，因为这个氟战我们不愿推迟，因为这个挑战我们志在必得，其
他的挑战也是如此。
```        

2. 多文字附加形状测试        

原图          
![2.png](https://upload-images.jianshu.io/upload_images/5617720-9c0225e2e176b3de.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

识别结果    
```text
2   We choose to go to the moon in this decade and do the
other things, not because they are easy, but because they are
hard, because that goal will serve to organize and measure
the best of our energies and skills, because that challenge is
one that we are willing to accept, one we are unwilling to
postpone, and one which we intend to win, and the others, too.

我们决定在这十年间登上月球并实现更多梦想，并非它们
轻而易举，而正是因为它们困难重重。因为这个目标将促进我
们实现最佳的组织并测试我们顶尖的技术和力量，因为这个挑
战我们乐于接受，因为这个挑战我们不愿推迟，因为这个挑战
我们志在必得，其他的挑战也是如此。
```        

**后序**      
经测试发现平均处理一张图片需要1~3s,对于以文字为主的图像识别效果尚可
对于有背景尤其是背景比较花哨的图片识别能力一般,较差. 目前的功能
满足对于类似于微博长文本图片情境下的处理及审核要求.

对于这种图片目前的代码几乎没有识别能力     
![dota.png](https://upload-images.jianshu.io/upload_images/5617720-f7aec2eda9978d84.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


有关改进与优化,这里主要思路一是首先将图片进行**降噪处理**,之后再进行ocr识别,       
二是**结合深度学习**有针对性的训练某一特定图片的OCR识别,      
*这里暂时没有查到相关的开源框架.暂时搁置.*

### 5. 视频言论聚合
这里主要是对上面的图片相关技术手段的补充,对于文本审核过的结果.
我们会根据他相关的视频进行聚合,当某一视频下面某类恶意文本产生过多时,我们将对其进行处理.(*具体处理方式?*)

## 流程

### 1. 色情图片<主动识别>
![image md5.png](https://upload-images.jianshu.io/upload_images/5617720-0f24fb1cdba9f37b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![pron](https://upload-images.jianshu.io/upload_images/5617720-bd899544398668bb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### 2. 其他恶意图片<被动相似>
![dHash](https://upload-images.jianshu.io/upload_images/5617720-1f81bb4eed01a403.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![sfit.png](https://upload-images.jianshu.io/upload_images/5617720-3b727aa01f9ced5b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/5617720-7cf76abbaddbda74.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 3. 人工审核
略

### 4. 总体流程
![all](https://upload-images.jianshu.io/upload_images/5617720-7ad31c04b4ae9787.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



## 策略 

### 审核策略
1. 上传的视频经过识别过审后才能播放
2. 点击量**上升速度**超过一定阈值的送人工审核接口

~~视频健康度~~
~~审核阈值~~

### 处理策略
* 色情视频/反动视频
  - 关键帧截图指纹加入恶意图库
  - 删除视频
  - 封禁视频上传者

* 低俗视频
  - 视频被锁定(无法新增点击量,点赞,分享且不会出现在首页及个性化推送中)
  - ~~视频上传者用户健康度扣20~~
