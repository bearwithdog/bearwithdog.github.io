---
layout: post
title: RefFaceInpainting
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

# RefFaceInpainting(换脸、换表情) 

## 重要参考项目   
[EXE-GAN](https://github.com/LonglongaaaGo/EXE-GAN?tab=readme-ov-file)   
在参考EXE-GAN项目换脸时候，在他的Notice中介绍了几个优秀的项目，后面介绍的内容是如何让Reference-Guided Face Inpainting运行起来。

## RefFaceInpainting介绍
[RefFaceInpainting](https://github.com/WuyangLuo/RefFaceInpainting)  
Acknowledgment中介绍了几个依赖的模型，在跑test时似乎只依赖[InsightFace-v2](https://github.com/foamliu/InsightFace-v2)， 这个模型是用来人脸对齐的，可以将图片上的人脸识别到，然后旋转到正的位置。其[releases](https://github.com/foamliu/InsightFace-v2/releases)中释放了几个预训练模型，BEST_checkpoint_r101.tar是RefFaceInpainting依赖的预训练模型。  

## 安装依赖的环境
```shell
conda create --name RefFaceInpainting python=3.8
conda activate RefFaceInpainting 
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge 
# 如果mkl版本大于2024.1会报错，可以用下面命令回退
#conda install mkl=2024.0 mkl-devel=2024.0 mkl-include=2024.0
pip install opencv-python 
pip install pyyaml==5.4.1 
pip install --upgrade git+https://github.com/Tramac/torchscope.git 
pip install scikit-image 
pip install torchsummary 
```

## 准备InsightFace-v2预训练模型  
直接使用BEST_checkpoint_r101.tar给RefFaceInpainting是不行的。(直接用BEST_checkpoint_r101.tar报错,多字段)  
1)因为作者保存的时候使用的nn.DataParallel保存的模型，模型会被封装进model字段中。
2)作者保存模型时候把膜性结构也封装进去了，但是RefFaceInpainting调用模型时候先创建模型，后只需要加载模型参数，所以给RefFaceInpainting的.PTH应该只包含模型权重就。
推荐搜索阅读:
**nn.DataParallel和torch.save结合保存和读取模型**
```python
        ....
        # RefFaceInpainting的trainer.py代码
        # arcface
28        self.arcface = resnet101() #创建模型结构
29        self.arcface.load_state_dict(torch.load('../BEST_checkpoint_r101.pth')) #只加载模型权重
        ...
```  
根据上面的说法，需要将.tar中的模型参数单独保存出来。
```python
import torch
from collections import OrderedDict

# 加载 checkpoint 文件
MODEL_FACE_RECOGNIZE_PATH = '/home/atlas/project/GAN/RefFaceInpainting/InsightFace-v2-1.0/BEST_checkpoint_r101.tar'
checkpoint = torch.load(MODEL_FACE_RECOGNIZE_PATH)

# 提取 state_dict
if 'model' in checkpoint:
    state_dict = checkpoint['model']
    # 检测下是不是torch.nn.DataParallel的实例，如果是那么对象中一定有module.state_dict()这样就可以提取到权重参数了
    if isinstance(state_dict, torch.nn.DataParallel):
        state_dict = state_dict.module.state_dict()
    elif hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
else:
    raise KeyError("'model' key not found in checkpoint")

# 创建新的有序字典，移除 'module.' 前缀（如果有）
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '')  # 去除 'module.' 前缀
    new_state_dict[name] = v

NEW_MODEL_PATH = '/home/atlas/project/GAN/RefFaceInpainting/InsightFace-v2-1.0/BEST_checkpoint_r101.pth'
torch.save(new_state_dict, NEW_MODEL_PATH)
print(f"Model weights have been saved to '{NEW_MODEL_PATH}'")
```
将上面代码复制，变成xx.py文件，将该文件放进InsightFace-v2的根目录下，这一步不可缺少(否则报错ModuleNotFoundError: No module named 'models')，或者在上面的代码上面加入下面两句话。
```python
import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/RefFaceInpainting/InsightFace-v2')
##看下面介绍
#https://github.com/pytorch/pytorch/issues/3678
##
```
原因:  
加载包含模型结构的 PyTorch 模型时，需要在项目文件夹中是因为 PyTorch 使用 Python 的 pickle 模块进行序列化和反序列化。pickle 模块会将模型的类和对象结构保存到文件中，当你加载模型时，pickle 需要能够找到原始的类定义。这意味着类定义必须在 Python 的模块搜索路径中。上面的操作就是要么把文件放入项目根目录让他自己搜索，要么就是把搜索路径手动加入。  

## 运行
得到新保存的BEST_checkpoint_r101.pth后就可以运行RefFaceInpainting项目，按github上的操作方法，下载RefFaceInpainting预训练模型model.pth和测试数据celebID。修改test.py中的model.pth的路径，还有trainer.py中的arcface读取BEST_checkpoint_r101.pth的路径，就可以运行了。
