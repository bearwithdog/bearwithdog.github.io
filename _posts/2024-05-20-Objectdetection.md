---
layout: post
title: 肿瘤自动识别的模型搭建
categories: algorithm
description: 
keywords: python, opencv ，机器学习
mermaid: false
sequence: false
flow: false
mathjax: false
mindmap: false
mindmap2: false
---
# 建模报告-自识别1.0  
## 数据来源  
青海单波长设备数据时间从202308150003--202312040003，共22人
## 数据提取逻辑
1 输入一侧乳房的全部多层数据    
2 不计算第一层和最后一层数据  
3 排除围挡区域，计算单层均值，取多层数据的最大均值  
4 对每一层数据进行单侧最大均值的归一化  
5 归一化后保留大于0.6的部分  
6 使用opencv提取边界  
7 计算边界围起来的以下标签值  
**0文件名 1宽 2高 3单层血红蛋白浓度均值 4单测血红蛋白浓度均值最大值 5可疑区域像素个数 6可疑区域血红蛋白总值 7可疑区域宽高比  8区域中心位置 9所属种类 10边界点**
## 特征萃取
根据以上提取出的特征再次萃取特征  
```python
file_path = r'E:\code\autocheck\autocheck\labels.txt'  # 替换为实际的文件路径
features = []
labels = []
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()  # 去除行尾的换行符和空白字符
        if line:  # 确保不处理空行
            data = line.split('#')  # 使用#分割每行的数据
            area = float(data[5])/(float(data[1])*float(data[2])) # 面积指数，感兴趣区域围起来的像素个数除以图像像素个数
            average = float(data[6])/float(data[5]) # 所围区域血红蛋白浓度均值
            relative  = average/float(data[4]) # 信号强度
            length_width_ratio = float(data[7]) # 长宽比
            features.append([area,average,relative,length_width_ratio])
            #features.append([average])
            if data[-2] == '1' or data[-2] == '2':
                labels.append(1) 
            else:
                labels.append(0) 
print(labels)
print(features)
```
仅保留[area,average,relative,length_width_ratio,x,y]6个特征
## 单一特征分析  
area
![area](/images/posts/Objectdetection/area.png)  
average
![average](/images/posts/Objectdetection/average.png)  
length_width_ratio
![length_width_ratio](/images/posts/Objectdetection/length_width_ratio.jpg)  
relative
![relative](/images/posts/Objectdetection/relative.jpg)   
单一特征，蓝色是伪影红色是肿瘤标记，从但标签看，区分度不是很好，我用皮尔森系数来度量了标签的相关性。遂采用下面的多特征分析，不同标签组合来看肿瘤和伪影的区分度。
## 多特征分析  
PCA  
area+average 可疑区域的面积+所围出来的均值
![PCA](/images/posts/Objectdetection/area_average.png)   
area+relative 可疑区域的面积+相对值
![PCA](/images/posts/Objectdetection/area_relative.png)   
所有特征
![PCA](/images/posts/Objectdetection/PCAall.png)   
使用多特征，然后使用PCA技术把多特征降为2维特征，发现数据分布上是有可分性的。  
可以看到在使用所有特征进行计算时候，不同种类有了更加聚合的趋势。蓝色点肿瘤特征会更加集中。 

## 模型搭建
使用所有特征加入分类器中训练模型，模型采用下面三个方案比较。  
我尝试了SVM支持向量机，决策树，逻辑回归。三个算法评比后选定逻辑回归来做这个分类，调参后的回归模型在测试集合上的表现。
![LR](/images/posts/Objectdetection/LR.jpg)   
逻辑回归可以清楚的给出预测的概率值，这点优于决策树（要不用XGBOOST?没试过）。

## 模型整合进代码进行回测
以下数据并未加入模型训练是陌生数据  
来自青海编号202401240001  
左侧  
![LR](/images/posts/Objectdetection/01.png)  
![LR](/images/posts/Objectdetection/02.png)  
![LR](/images/posts/Objectdetection/03.png)  
左侧全部判定为伪影，图片上的数字是来自模型预测肿瘤的概率值，目前只是2分类
右侧    
![LR](/images/posts/Objectdetection/10.png)  
![LR](/images/posts/Objectdetection/11.png)  
![LR](/images/posts/Objectdetection/12.png)  
病理对照  
![LR](/images/posts/Objectdetection/bl.png)  
结果还可以  
## 结论  
模型是在既往数据上寻找规律来预测未来的数据，所以前置假设就是设备产生的数据是独立同分布的，虽然在建模过程中，我们采用了标准化的操作，通过变换将数据标准化可以减缓一些绝对值漂移，但是良好稳定的数据是模型发挥更好效果的前置。  
项目代码：
https://gitee.com/liutengyu1989/autocheck/tree/test01/  