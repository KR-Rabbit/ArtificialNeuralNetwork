# 基于numpy实现的人工神经网络

## 1. 项目结构

├── README.MD
├── data:MINST数据集
├── model:网络结构定义
├── result:训练结果
├── utils:工具类,数据加载，模型验证，结果保存等。
└── train.py:训练主程序

## 2. 项目运行

运行train.py即可，训练结果保存在result文件夹下

## 3. 理论知识

1. ### 交叉熵损失函数

   交叉熵损失函数是分类问题中常用的损失函数，其定义如下：
   $$
   L=-\sum_{i=1}^{n}y_{i}log\hat{y_{i}}
   $$

2. ### Softmax

   Softmax函数是一个常用的激活函数，其定义如下：
   $$
   \sigma(z)_{j}=\frac{e^{z_{j}}}{\sum_{k=1}^{K}e^{z_{k}}}
   $$
3. ### -

   梯度是一个向量，其方向是函数值增加最快的方向，其大小是函数值增加最快的速率。
   $$
   \nabla f(x)
   =\left[\frac{\partial f}{\partial x_{1}},\frac{\partial f}{\partial x_{2}},\cdots,\frac{\partial f}{\partial x_{n}}\right]
   $$
   Softmax函数的导数为
   $$
   f'(x)=f(x)(1-f(x))
   $$
   交叉熵损失对于Softmax输出的梯度为
   $$
   \frac{\partial L}{\partial z_{j}}=\hat{y_{j}}-y_{j}
   $$
   对于z=w*x+b，z对w梯度：
   $$
   \frac{\partial z}{\partial w_{j}}=x_{j}
   $$
   即在人工神经网络中，最后的梯度是损失函数对于输出层的梯度，即$y_{j}-\hat{y_{j}}$。