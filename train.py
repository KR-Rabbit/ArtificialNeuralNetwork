import random

import numpy as np
from matplotlib import pyplot as plt

from model import NeuralNetwork
from utils.data import MNIST, save_data, show_save_fig

mnist = MNIST(root="./data", transform=[lambda x: x / 255.0,
                                        lambda x: x.reshape(x.shape[0], -1)])  # transform 原始数据 范围0-255,shape 28x28
# 数据集
train_images, train_labels = mnist.train_images, mnist.train_labels
test_images, test_labels = mnist.test_images, mnist.test_labels

# 可视化,随机显示16张图片
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(train_images[random.randint(0, len(train_images))], cmap="gray")
    plt.title(f"Label: {train_labels[random.randint(0, len(train_labels))]}")
    plt.axis("off")
plt.show()
# 超参数
lr = 1e-1
epochs = 50
batch_size = 64
hidden_layer_num = 3
hidden_size = [256, 128, 64]

net = NeuralNetwork(in_size=784, out_size=10, hidden_layer_num=hidden_layer_num, hidden_size=hidden_size)

loss_list = []
accuracy_list = []
for epoch in range(epochs):
    batch_index = 0
    tmp_loss = []
    while batch_index < len(train_images):  # 小批量梯度下降
        batch_x = train_images[batch_index: batch_index + batch_size]
        batch_y = train_labels[batch_index: batch_index + batch_size]
        batch_index += batch_size
        # forward
        y_pred = net.forward(batch_x)
        # backward
        net.backward(batch_y, y_pred)
        # update
        net.update(lr)
        # loss
        tmp_loss.append(net.loss(batch_x, batch_y))
    loss_list.append(np.mean(tmp_loss).item())
    accuracy_list.append(net.accuracy(test_images, test_labels))
    print(f"Epoch: {epoch + 1}, Loss: {loss_list[-1]}, Accuracy: {accuracy_list[-1]}")
# 保存数据
save_data(loss_list, "result/loss.txt")
save_data(accuracy_list, "result/accuracy.txt")
# 可视化并保存
show_save_fig(loss_list, y_label="loss", title="Loss", save_path="result/loss.png")
show_save_fig(accuracy_list, y_label="accuracy", title="Accuracy", save_path="result/accuracy.png")
