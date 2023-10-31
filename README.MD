<h1>基于numpy实现的人工神经网络</h1>

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
   <img src="images/CrossEntropyLoss.png">

2. ### Sigmoid

   Sigmoid函数是一个常用的激活函数，其定义如下：
   <img src="images/Softmax.png">
3. ### -

   梯度是一个向量，其方向是函数值增加最快的方向，其大小是函数值增加最快的速率。
   <img src="images/gradient.png">
   Sigmoid函数的导数为
   <img src="images/grad_softmax.png">
   交叉熵损失对于Softmax输出的梯度为
   <img src="images/grad_loss.png">

  