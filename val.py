import argparse

import numpy as np

from model import NeuralNetwork
from utils import get_root, increment_path
from utils.confusion_matrix import *
from utils.data import MNIST

ROOT = get_root(__file__, 1)


def run(data,
        net=None,
        bath_size=64,
        save_dir=ROOT / "logs/val",
        console=False,
        file=False,
        ):
    # 数据集
    if isinstance(data, MNIST):
        mnist = data

    else:
        mnist = MNIST(root=data, transform=[lambda x: x / 255.0,
                                            lambda x: x.reshape(x.shape[0], -1)])  # transform 原始数据 范围0-255,shape 28x28
    test_images, test_labels = mnist.test_images, mnist.test_labels
    # 加载模型
    training = net is not None
    if not training:
        net_path = net
        net = NeuralNetwork()
        net.load(net_path)

    net.detect()
    # 测试
    y_trues = []
    y_preds = []
    batch_index = 0
    mv_loss = 0.0
    while batch_index < len(test_images):
        batch_x = test_images[batch_index: batch_index + bath_size]
        batch_y = test_labels[batch_index: batch_index + bath_size]
        batch_index += bath_size
        y_trues.extend(batch_y)
        y_pred = net(batch_x)
        loss = net.loss(y_pred, batch_y)
        mv_loss = (mv_loss * batch_index + loss) / (batch_index + 1)
        y_preds.extend(np.argmax(y_pred, axis=1))
    # 转为一维数组
    y_trues = np.array(y_trues).reshape(-1)
    y_preds = np.array(y_preds).reshape(-1)
    # 混淆矩阵
    matrix = get_confusion_matrix(y_trues, y_preds, labels=[i for i in range(10)])
    # 评估 acc, precision, recall, f1
    scores = get_score(matrix)
    if not training:
        save_dir = increment_path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if console or file:
            log_scores(scores, console=console, file=file, save_dir=save_dir)
        plot_confusionn_matrix(matrix, save_dir=save_dir, names=[i for i in range(10)])
    return scores, mv_loss  # 返回评估结果


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data", help="data root")
    parser.add_argument("--net", type=str, required=True, help="net params path")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--console", action="store_true", help="print result of evaluation")
    parser.add_argument("--file", action="store_true", help="save result of evaluation to txt")
    parser.add_argument("--save_dir", type=str, default="./result", help="save dir")
    return parser.parse_args()


def main(opt):
    main(**vars(opt))


if __name__ == '__main__':
    opt = parser_opt()  # 从命令行获取参数
    main(opt)
