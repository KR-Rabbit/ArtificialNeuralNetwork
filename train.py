import random

import numpy as np
from matplotlib import pyplot as plt

from model import NeuralNetwork
from utils import test
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
    img_idx = random.randint(0, len(train_images))
    plt.imshow(train_images[img_idx].reshape(28, 28), cmap="gray")
    plt.title(f"Label: {train_labels[img_idx]}")
    plt.axis("off")
plt.show()
# 超参数
lr = 1e-1
epochs = 50
batch_size = 64
hidden_layer_num = 3
hidden_size = [256, 128, 64]

net = NeuralNetwork(in_size=784, out_size=10, hidden_layer_num=hidden_layer_num, hidden_size=hidden_size)

train_loss_list = []
test_loss_list = []
accuracy_list = []
for epoch in range(epochs):
    batch_index = 0
    tmp_train_loss = []
    tmp_test_loss = []
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
        tmp_train_loss.append(net.loss(batch_x, batch_y))
    train_loss_list.append(np.mean(tmp_train_loss).item())
    l, acc = test(net, test_images, test_labels)
    test_loss_list.append(l)
    accuracy_list.append(acc)

    # 验证
    l, acc = test(net, test_images, test_labels)

    print(f"Epoch: {epoch + 1}, Loss: {train_loss_list[-1]:.4f}, Accuracy: {accuracy_list[-1]:.4f}")

net.save("result/params.pkl")
# 保存数据
save_data(train_loss_list, "result/train_loss.txt")
save_data(test_loss_list, "result/test_loss.txt")
save_data(accuracy_list, "result/accuracy.txt")
# 可视化并保存
show_save_fig(train_loss_list, y_label="loss", title="Train Loss", save_path="result/train_loss.png")
show_save_fig(test_loss_list, y_label="loss", title="Test Loss", save_path="result/test_loss.png")
show_save_fig(accuracy_list, y_label="accuracy", title="Accuracy", save_path="result/accuracy.png")
