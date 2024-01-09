import pathlib
from typing import Iterable

import numpy as np


class NeuralNetwork:
    def __init__(self, in_size=784, out_size=10, hidden_layer_num=2, hidden_size=None, detect=False):
        self.__cache = None  # 缓存，记录每层的输出，用于反向传播
        self.__grads = None  # 权重梯度，记录每层的梯度
        self.__bias = None  # 偏置
        self.__weights = None  # 权重，字典类型，记录每层的权重
        self.__in_size = in_size  # 输入特征的个数
        self.__out_size = out_size  # 输出特征的个数
        self.__hidden_layer_num = hidden_layer_num  # 隐藏层数
        self.__hidden_size = hidden_size  # 隐藏层的神经元个数
        self.check()
        self.__detect_mode = detect
        if not detect:  # 如果不是检测模式，就随机初始化权重
            self.train()

    def train(self):
        self.__weights = {f"w{i + 1}": np.random.randn(s1, s2) for i, (s1, s2) in
                          enumerate(zip([self.__in_size] + self.__hidden_size,
                                        self.__hidden_size + [self.__out_size]))}  # 权重，字典类型，记录每层的权重
        self.__bias = {f"b{i + 1}": np.random.randn(s) for i, s in
                       enumerate(self.__hidden_size + [self.__out_size])}  # 偏置
        self.__grads = {f"w{i + 1}": np.zeros_like(w) for i, w in
                        enumerate(self.__weights.values())}  # 权重梯度，记录每层的梯度
        self.__grads.update({f"b{i + 1}": np.zeros_like(b) for i, b in enumerate(self.__bias.values())})
        self.__cache = {}  # 缓存，记录每层的输出，用于反向传播

    def detect(self):
        self.__detect_mode = True

    def check(self):
        if self.__hidden_size is None:
            self.__hidden_size = [self.__in_size // 2] * self.__hidden_layer_num  # 隐藏层的神经元个数
        elif isinstance(self.__hidden_size, int):
            self.__hidden_size = [self.__hidden_size] * self.__hidden_layer_num
        elif isinstance(self.__hidden_size, Iterable):
            if hasattr(self.__hidden_size, "__len__") and len(self.__hidden_size) == self.__hidden_layer_num:
                self.__hidden_size = self.__hidden_size
            else:
                self.__hidden_size = [self.__in_size // (2 ** i) for i in range(self.__hidden_layer_num)]
        else:
            raise TypeError("hidden_size must be int or Iterable")

    def forward(self, x):  # 前向传播
        if self.__detect_mode:
            for i in range(self.__hidden_layer_num):
                w_i_1 = self.__weights[f"w{i + 1}"]
                b_i_1 = self.__bias[f"b{i + 1}"]
                z = np.dot(x, w_i_1) + b_i_1
                x = self.sigmoid(z)
            w_last = self.__weights[f"w{self.__hidden_layer_num + 1}"]
            b_last = self.__bias[f"b{self.__hidden_layer_num + 1}"]
            z = np.dot(x, w_last) + b_last
            return self.softmax(z)
        self.__cache["a0"] = x  # a0 = x
        for i in range(self.__hidden_layer_num):
            a_i = self.__cache[f"a{i}"]  # 上一层的输出，作为这一层的输入
            w_i_1 = self.__weights[f"w{i + 1}"]
            b_i_1 = self.__bias[f"b{i + 1}"]
            z = np.dot(a_i, w_i_1) + b_i_1  # z = w*x + b
            a = self.sigmoid(z)  # a = sigmoid(z)
            self.__cache[f"z{i + 1}"] = z
            self.__cache[f"a{i + 1}"] = a
        a_pre_last = self.__cache[f"a{self.__hidden_layer_num}"]  # 倒数第二层的输出
        w_last = self.__weights[f"w{self.__hidden_layer_num + 1}"]
        b_last = self.__bias[f"b{self.__hidden_layer_num + 1}"]

        self.__cache[f"z{self.__hidden_layer_num + 1}"] = np.dot(a_pre_last, w_last) + b_last
        self.__cache[f"a{self.__hidden_layer_num + 1}"] = self.softmax(self.__cache[f"z{self.__hidden_layer_num + 1}"])
        return self.__cache[f"a{self.__hidden_layer_num + 1}"]  # shape (batch-size,10)

    def backward(self, y, y_pred):  # 反向传播
        # delta = y_pred - y
        y = self.one_hot_encode(y)
        delta = y_pred - y  # dJ/da = y_pred - y 交叉熵损失对于a的导数为y_pred - y
        self.__grads[f"w{self.__hidden_layer_num + 1}"] = np.dot(self.__cache[f"a{self.__hidden_layer_num}"].T,
                                                                 delta) / len(y)
        self.__grads[f"b{self.__hidden_layer_num + 1}"] = np.sum(delta, axis=0) / len(y)
        # 计算前面的梯度
        pre_grad = delta
        for i in range(self.__hidden_layer_num, 0, -1):
            pre_grad = np.dot(pre_grad, self.__weights[f"w{i + 1}"].T) * self.__cache[f"a{i}"] * (
                    1 - self.__cache[f"a{i}"])
            self.__grads[f"w{i}"] = np.dot(self.__cache[f"a{i - 1}"].T, pre_grad)
            self.__grads[f"b{i}"] = np.sum(pre_grad, axis=0)

    def update(self, lr):  # 更新参数
        for i in range(1, self.__hidden_layer_num + 1):
            self.__weights[f"w{i}"] -= lr * self.__grads[f"w{i}"]
            self.__bias[f"b{i}"] -= lr * self.__grads[f"b{i}"]

    def one_hot_encode(self, y):  # one-hot编码
        return np.eye(self.__out_size)[y]  #

    @staticmethod
    def one_hot_decode(y):  # one-hot解码
        return np.argmax(y, axis=1)

    @staticmethod
    def sigmoid(x):  # sigmoid函数
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):  # softmax函数
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def loss(self, x, y_true):
        y = self.one_hot_encode(y_true)
        return -np.sum(y * np.log(self.forward(x))) / len(x)

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def save(self, path):
        path = pathlib.Path(path)
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True)
        with path.open("wb") as f:
            # 保存权重，输入维度，输出维度，隐藏层数，隐藏层维度
            np.savez(f, **self.__weights, **self.__bias, in_size=self.__in_size, out_size=self.__out_size,
                     hidden_layer_num=self.__hidden_layer_num, hidden_size=self.__hidden_size)

    # 加载权重
    def load(self, path):
        path = pathlib.Path(path)
        assert path.exists(), f"{path} not exists"
        with path.open("rb") as f:
            data = np.load(f)
            self.__weights = {k: v for k, v in data.items() if k.startswith("w")}
            self.__bias = {k: v for k, v in data.items() if k.startswith("b")}
            self.__in_size = data["in_size"]
            self.__out_size = data["out_size"]
            self.__hidden_layer_num = data["hidden_layer_num"]
            self.__hidden_size = data["hidden_size"]
