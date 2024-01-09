import warnings
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn


# # TP TN FP FN，多标签分类
# def score(y_true, y_pred):
#     TP = ((y_true == 1) & (y_pred == 1)).sum()
#     TN = ((y_true == 0) & (y_pred == 0)).sum()
#     FP = ((y_true == 0) & (y_pred == 1)).sum()
#     FN = ((y_true == 1) & (y_pred == 0)).sum()
#     return TP, TN, FP, FN


# 混淆矩阵
def con_matrix(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)


# 绘制混淆矩阵
def plot(matrix, normalize=True, save_dir='', names=()):
    array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    nc = len(names)  # number of classes, names
    sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
    labels = 0 < nc < 99  # apply names to ticklabels
    ticklabels = names if labels else "auto"
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
        sn.heatmap(array,
                   ax=ax,
                   annot=nc < 30,
                   annot_kws={
                       "size": 8},
                   cmap='Blues',
                   fmt='.2f',
                   square=True,
                   vmin=0.0,
                   xticklabels=ticklabels,
                   yticklabels=ticklabels).set_facecolor((1, 1, 1))
    ax.set_ylabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix')
    fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
    plt.close(fig)


# 根据混淆矩阵计算acc, precision,recall,f1
def score(matrix):
    matrix = matrix.astype(np.float32)
    # 计算矩阵的对角线元素
    diag = np.diag(matrix).astype(np.float32)
    # 计算每一列的和
    sum_col = matrix.sum(axis=0).astype(np.float32)
    # 计算每一行的和
    sum_row = matrix.sum(axis=1).astype(np.float32)
    # 计算精确率
    precision = diag / sum_col
    # 计算召回率
    recall = diag / sum_row
    # 计算F1值
    f1 = 2 * precision * recall / (precision + recall)
    # 计算acc
    acc = diag.sum() / matrix.sum()
    return acc, precision, recall, f1


if __name__ == '__main__':


    # 测试集
    y_trues = []
    y_preds = []

    # 10分类，100个样本
    y_true = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 0, 0, 0, 8, 8, 9, 9, 9, 9])
    # model(x_data) --->y_pred
    y_pred = np.array([1, 1, 2, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 4, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 6, 7, 8, 8, 8, 8, 0, 0, 0, 8, 8, 9, 9, 9, 9])
    matrix = con_matrix(y_true, y_pred, labels=[i for i in range(10)])
    plot(matrix, names=[i for i in range(10)])
    acc, precision, recall, f1 = score(matrix)
    # 格式化标头，acc,precision,recall,对齐
    formats_1 = "{:<10}" + "{:>14}" * 10
    formats_2 = "{:<10}" + "{:>14.2f}" * 10
    strings = [f"Label-{i}" for i in range(1, 11)]
    print(formats_1.format(" ", *strings))
    print(formats_2.format("precision", *precision))
    print(formats_2.format("recall", *recall))
    print(formats_2.format("f1", *f1))
    print("{:<10}{:>14.2f}".format("acc-all", acc))
