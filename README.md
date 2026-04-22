# GoogLeNet (Inception V1)
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result
- GoogLeNet training (Backbone Only):
<img width="2480" height="1914" alt="googlenet_training_curve" src="https://github.com/user-attachments/assets/32a4819a-8e1c-476c-b243-356546bbbce9" />

- GoogLeNet:


由于网络非常深，训练非常耗时，仅以20轮作为观看，但其并未完全收敛，可自行增大epoch，让其完全收敛进行测试，这里结果仅对比带上辅助分类器的效果（未来的网络其实已取消了这个功能，仅作为复现学习思维模型），辅助分类器是为了解决梯度消失问题的，但其实加上也可能会导致出现梯度消失问题，所以GoogLeNet对于调参来说非常重要。

---

## 简介
GoogLeNet（又称Inception V1）是由Christian Szegedy、Wei Liu、Yangqing Jia等人于2014年提出的里程碑式深度卷积神经网络，相关成果发表于《Going Deeper with Convolutions》，并斩获ILSVRC 2014图像分类竞赛冠军。它突破了当时单纯通过加深网络提升性能的主流思路，首次提出**Inception多分支模块**与**辅助分类器**两大核心设计，通过多尺度特征融合大幅提升了特征提取效率，同时有效缓解了深度网络的梯度消失问题。GoogLeNet将ImageNet分类错误率降至6.67%，参数量仅为AlexNet的1/12，开创了高效多分支网络的先河，为后续Inception系列、ResNet、MobileNet等经典模型奠定了关键技术基础。

## 架构
GoogLeNet的核心架构为**多分支堆叠式深度神经网络**，整体由**前期卷积层**、**9个串联的Inception特征提取模块**、**2个辅助分类器**与**全局平均池化分类模块**构成，大幅减少了全连接层的使用。原论文标准输入为224×224分辨率的3通道RGB图像，最终输出对应分类类别的预测概率，具体结构与设计如下：
- **特征提取核心（Inception模块）**：每个Inception模块包含4个并行分支，分别使用1×1、3×3、5×5卷积核与3×3最大池化提取不同尺度的特征，所有分支输出在通道维度拼接。其中1×1卷积用于降维减少计算量，解决了大卷积核带来的参数量爆炸问题。
- **梯度缓解机制（辅助分类器）**：在网络中间层（Inception 4a和4d后）添加两个轻量级辅助分类器，训练时将其输出按权重（原论文为0.3）加入总损失，为底层网络提供额外的梯度信号，有效缓解了深度网络的梯度消失问题。
- **分类输出模块（全局平均池化）**：最后一个Inception模块输出后使用全局平均池化将每个通道的特征图平均为一个标量，替代传统的全连接层分类头，大幅减少了参数量并显著降低了过拟合风险。

该架构通过多尺度特征融合实现了高效的特征表达，在保持高性能的同时极大地降低了计算复杂度，其"用计算效率换网络深度"的设计思想成为了现代深度学习模型的重要准则。

<img width="1122" height="2983" alt="image" src="https://github.com/user-attachments/assets/77ed8ef8-2178-44e9-ac75-c474c59228cf" />
<img width="414" height="625" alt="image" src="https://github.com/user-attachments/assets/37c3eaa9-7513-4466-b9f1-c075d6664e4e" />
<img width="839" height="288" alt="image" src="https://github.com/user-attachments/assets/c2502f38-b384-4849-b1df-4671685e0904" />

如果你想看更详细的模型解答可以观看：https://developer.aliyun.com/article/1062593

**注意**：我们使用的是数据集CIFAR-10，它是10类数据，并且不同于原文献，由于CIFAR-10图像尺寸（32×32）远小于原论文的224×224，我们会对网络结构做微小适配（主要是调整前几层卷积核和步长，移除过早的池化层，修改辅助分类器全连接层输入维度），但核心架构（Inception多分支模块 + 辅助分类器 + 全局平均池化）完全保留。

## 数据集
我们使用的是数据集CIFAR-10，是一个更接近普适物体的彩色图像数据集。CIFAR-10是由Hinton的学生Alex Krizhevsky和Ilya Sutskever整理的一个用于识别普适物体的小型数据集。一共包含10个类别的RGB彩色图片：飞机（airplane）、汽车（automobile）、鸟类（bird）、猫（cat）、鹿（deer）、狗（dog）、蛙类（frog）、马（horse）、船（ship）和卡车（truck）。每个图片的尺寸为32×32，每个类别有6000个图像，数据集中一共有50000张训练图片和10000张测试图片。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

它不同于我们常见的图片存储格式，而是用二进制优化了储存，当然我们也可以将其复刻出来为PNG等图片格式，但那会很大，我们的目标是神经网络，这里不做细致解析数据集，如果你想了解该数据集请观看链接：https://cloud.tencent.com/developer/article/2150614


---

## Introduction
GoogLeNet (also known as Inception V1) is a landmark deep convolutional neural network proposed by Christian Szegedy, Wei Liu, Yangqing Jia et al. in 2014. Its findings were published in "Going Deeper with Convolutions" and won the ILSVRC 2014 Image Classification Competition. It broke through the mainstream idea of improving performance simply by deepening the network at that time, and first proposed two core designs: **Inception multi-branch module** and **auxiliary classifier**. It greatly improved the efficiency of feature extraction through multi-scale feature fusion, and effectively alleviated the gradient vanishing problem of deep networks. GoogLeNet reduced the ImageNet classification error rate to 6.67% with only 1/12 the number of parameters of AlexNet, pioneered efficient multi-branch networks, and laid a key technical foundation for subsequent classic models such as the Inception series, ResNet, and MobileNet.

## Architecture
The core architecture of GoogLeNet is a **multi-branch stacked deep neural network**, which is composed of **preliminary convolutional layers**, **9 serial Inception feature extraction modules**, **2 auxiliary classifiers** and a **global average pooling classification module**, which greatly reduces the use of fully connected layers. The original paper's standard input was a 224×224 resolution 3-channel RGB image, and the final output was the predicted probability of the corresponding classification category. The specific structure and design are as follows:

- **Core Feature Extraction (Inception Module)**: Each Inception module contains 4 parallel branches, which use 1×1, 3×3, 5×5 convolution kernels and 3×3 max pooling to extract features of different scales respectively. The outputs of all branches are concatenated along the channel dimension. Among them, 1×1 convolution is used for dimensionality reduction to reduce the amount of calculation and solve the problem of parameter explosion caused by large convolution kernels.
- **Gradient Mitigation Mechanism (Auxiliary Classifier)**: Two lightweight auxiliary classifiers are added in the middle layers of the network (after Inception 4a and 4d). During training, their outputs are added to the total loss with a weight (0.3 in the original paper), providing additional gradient signals to the lower layers of the network and effectively alleviating the gradient vanishing problem of deep networks.
- **Classification Output Module (Global Average Pooling)**: After the output of the last Inception module, global average pooling is used to average the feature map of each channel into a scalar, replacing the traditional fully connected layer classification head, which greatly reduces the number of parameters and significantly reduces the risk of overfitting.

This architecture achieves efficient feature expression through multi-scale feature fusion, and greatly reduces computational complexity while maintaining high performance. Its design idea of "trading computational efficiency for network depth" has become an important criterion for modern deep learning models.

<img width="1122" height="2983" alt="image" src="https://github.com/user-attachments/assets/eee194b0-6b69-4be4-adc0-0f41a816031c" />
<img width="724" height="251" alt="image" src="https://github.com/user-attachments/assets/f7a7aa9a-b9f1-40b6-895c-d412de44a8c1" />
<img width="824" height="275" alt="image" src="https://github.com/user-attachments/assets/f3625d95-7a4f-4d5e-865a-6839600bcd25" />



**Note:** We use the CIFAR-10 dataset, which is a 10-class dataset. Unlike the original paper, the image size of CIFAR-10 (32×32) is much smaller than the 224×224 in the original paper. We will make minor adaptations to the network structure (mainly adjusting the convolution kernels and strides of the first few layers, removing premature pooling layers, and modifying the input dimensions of the auxiliary classifier's fully connected layers), but the core architecture (Inception multi-branch module + auxiliary classifier + global average pooling) will be completely retained.

## Dataset
We used the CIFAR-10 dataset, a color image dataset that more closely approximates common objects. CIFAR-10 is a small dataset for recognizing common objects, compiled by Hinton's students Alex Krizhevsky and Ilya Sutskever. It contains RGB color images for 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32 × 32 pixels, with 6000 images per category. The dataset contains 50,000 training images and 10,000 test images.

The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

It differs from common image storage formats, using binary-optimized storage. While we could recreate it as PNG or other image formats, that would result in a very large file size. Our focus is on neural networks, so we won't delve into a detailed analysis of the dataset here. If you'd like to learn more about this dataset, please see the link: https://cloud.tencent.com/developer/article/2150614

---
## 原文章 | Original article
Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
