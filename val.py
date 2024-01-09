import argparse

import numpy as np

from model import NeuralNetwork
from utils.confusion_matrix import con_matrix, score, plot
from utils.data import MNIST


def main(opt):
    # 数据集
    mnist = MNIST(root=opt.data, transform=[lambda x: x / 255.0,
                                            lambda x: x.reshape(x.shape[0], -1)])  # transform 原始数据 范围0-255,shape 28x28
    test_images, test_labels = mnist.test_images, mnist.test_labels

    # 加载模型
    net = NeuralNetwork()
    net.detect()
    net.load("./result/params.pkl")

    # 测试
    y_trues = []
    y_preds = []
    bath_size = 64
    batch_index = 0
    while batch_index < len(test_images):
        batch_x = test_images[batch_index: batch_index + bath_size]
        batch_y = test_labels[batch_index: batch_index + bath_size]
        batch_index += bath_size
        y_trues.extend(batch_y)
        y_preds.extend(net.predict(batch_x))
    # 转为一维数组
    y_trues = np.array(y_trues).reshape(-1)
    y_preds = np.array(y_preds).reshape(-1)
    # 混淆矩阵
    matrix = con_matrix(y_trues, y_preds, labels=[i for i in range(10)])
    # 评估
    acc, precision, recall, f1 = score(matrix)
    if opt.print:
        # 格式化标头，acc,precision,recall,对齐
        formats_1 = "{:<10}" + "{:>14}" * 10
        formats_2 = "{:<10}" + "{:>14.2f}" * 10
        strings = [f"Label-{i}" for i in range(1, 11)]
        print(formats_1.format(" ", *strings))
        print(formats_2.format("precision", *precision))
        print(formats_2.format("recall", *recall))
        print(formats_2.format("f1", *f1))
        print("{:<10}{:>14.2f}".format("acc-all", acc))
    plot(matrix, save_dir=opt.save_dir, names=[i for i in range(10)])


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data", help="data root")
    parser.add_argument("--print", action="store_true", help="print result of evaluation")
    parser.add_argument("--save_dir", type=str, default="./result", help="save dir")
    return parser.parse_args()


def run(**kwargs):
    opt = parser_opt()
    opt.__dict__.update(kwargs)
    main(opt)


if __name__ == '__main__':
    opt = parser_opt()
    main(opt)
