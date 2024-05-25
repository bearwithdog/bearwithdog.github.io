---
layout: post
title: 红外相机乳房轮廓数据提取 
categories: algorithm
description: 
keywords: python, opencv , 深度学习
mermaid: false
sequence: false
flow: false
mathjax: false
mindmap: false
mindmap2: false
---

# 红外相机的乳房轮廓图像分割融合

## 问题难点   
1 光源是红外光而且光源下置导致光照不稳定，如下图  
![1](/images/posts/Image_Segmentation/1.png)    
2 因为原因1导致传统边缘检测算法鲁棒性很差  

## 解决方案  
首先2问题依赖1问题，所以不管采用传统的图像还是结合深度学习的图像分割，都需要解决。  
解决方案也很常见，直方图均衡。关于该算法原理就不赘述，网上很多写的还非常清楚，原理就是原始图像的像素值分布集中在某一块区域所以导致边缘不清晰，直方图均衡就是将原始分布拉伸到更宽的范围，这样像素的数值变化就明显了，边界自然看清。  
但是在使用opencv时候直接调用直方图均衡有点问题--opencv直方图均衡函数是针对8位图像来写的。但是设备中用的红外相机，他的图像深度是16位，如果将原来图像转成8位再使用opencv函数来均衡，效果不好。如下（我做了一些处理采用的是局部均衡化拼接）：  
![2](/images/posts/Image_Segmentation/2.png)  
如果这样不行的话，也很简单那就找一个直方图均衡的原始代码，改一下，自己做一个16位的图像处理函数。
```python
#coding:utf-8
#*********************************************************************************************************
'''
说明：利用python/numpy/opencv实现直方图均衡化，其主要思想是将一副图像的直方图分布变成近似均匀分布，从而增强图像的对比度
算法思路:
        1)以灰度图的方式加载图片;
        2)求出原图的灰度直方图，计算每个灰度的像素个数在整个图像中所占的百分比;
		3)计算图像各灰度级的累积概率密度分布；
		4)求出新图像的灰度值。
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Origin_histogram( img ):
    #建立原始图像各灰度级的灰度值与像素个数对应表
    histogram = {}
    for i in range( img.shape[0] ):
        for j in range( img.shape[1] ):
            k = img[i][j]
            if k in histogram:
                histogram[k] += 1
            else:
                histogram[k] = 1
                
    sorted_histogram = {}#建立排好序的映射表
    sorted_list = sorted( histogram )#根据灰度值进行从低至高的排序
    
    for j in range( len( sorted_list ) ):
        sorted_histogram[ sorted_list[j] ] = histogram[ sorted_list[j] ]

    return sorted_histogram

def equalization_histogram( histogram, img ):
    
    pr = {}#建立概率分布映射表
    
    for i in histogram.keys():
        pr[i] = histogram[i] / ( img.shape[0] * img.shape[1] ) 

    tmp = 0
    for m in pr.keys():
        tmp += pr[m]
        pr[m] =  max( histogram ) * tmp
    
    new_img = np.zeros( shape = ( img.shape[0], img.shape[1] ), dtype = np.uint8 )
    
    for k in range( img.shape[0] ):
        for l in range( img.shape[1] ):
            new_img[k][l] = pr[img[k][l]]
            
    return new_img


def GrayHist( img ):
    # 计算灰度直方图
    height, width = img.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(height):
        for j in range(width):
            grayHist[img[i][j]] += 1
    return grayHist
    
if __name__ == '__main__':
    #读取原始图像
    img = cv2.imread( 'lowlight.png', cv2.IMREAD_GRAYSCALE )
    #计算原图灰度直方图
    origin_histogram = Origin_histogram( img )
    #直方图均衡化
    new_img = equalization_histogram( origin_histogram, img )
    
    origin_grayHist = GrayHist(img)
    equaliza_grayHist = GrayHist( new_img )
    x = np.arange(256)
    # 绘制灰度直方图
    plt.figure( num = 1 )
    plt.subplot( 2, 2, 1 )
    plt.plot(x, origin_grayHist, 'r', linewidth=2, c='black')
    plt.title("Origin")
    plt.ylabel("number of pixels")
    plt.subplot( 2, 2, 2 )
    plt.plot(x, equaliza_grayHist, 'r', linewidth=2, c='black')
    plt.title("Equalization")
    plt.ylabel("number of pixels")
    plt.subplot( 2, 2, 3 )
    plt.imshow( img, cmap = plt.cm.gray )
    plt.title( 'Origin' )
    plt.subplot( 2, 2, 4 )
    plt.imshow( new_img, cmap = plt.cm.gray )
    plt.title( 'Equalization' )
    plt.show()
```
只需要将上面代码逻辑中图像范围[256]->[65535]。  
![3](/images/posts/Image_Segmentation/3.png)   
有了这个结果图像的前处理就完成了。  
后面自然的就是数据标注->训练->模型导出onnx->推理嵌入主要流程中，属于工程开发了。  
数据标注（差不多200多张）  
![4](/images/posts/Image_Segmentation/4.png)
最终效果，轮廓采集到，然后和重构的图像进行融合，乳房轮廓可以看到了。 
![5](/images/posts/Image_Segmentation/5.png)

项目代码：
https://gitee.com/liutengyu1989/breast_outlinev4 