import warnings
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn


# 混淆矩阵
def get_confusion_matrix(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)


# 绘制混淆矩阵
def plot_confusionn_matrix(matrix, normalize=True, save_dir='', names=()):
    array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True, dpi=1000)
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
                       "size": 10},
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
def get_score(matrix):
    matrix = matrix.astype(np.float32)
    # 计算矩阵的对角线元素
    diag = np.diag(matrix).astype(np.float32)
    # 计算每一列的和
    sum_col = matrix.sum(axis=0).astype(np.float32)
    # 计算每一行的和
    sum_row = matrix.sum(axis=1).astype(np.float32)
    # 计算精确率
    sum_col[sum_col == 0] = 1e-10  # 防止除0
    sum_row[sum_row == 0] = 1e-10
    precision = diag / sum_col
    # 计算召回率
    recall = diag / sum_row
    # 计算F1值
    precision[precision == 0] = 1e-10 # 防止除0
    f1 = 2 * precision * recall / (precision + recall)
    # 计算acc
    acc = diag.sum() / matrix.sum()
    return acc, precision, recall, f1


def log_scores(scores: tuple, console=False, file=False, save_dir='.'):
    if not console and not file:  # 不打印到控制台，不保存到文件
        return
    acc, precision, recall, f1 = scores
    formats_1 = "{:<10}" + "{:>14}" * 10
    formats_2 = "{:<10}" + "{:>14.2f}" * 10
    strings = [f"Label-{i}" for i in range(1, 11)]
    contents = [
        formats_1.format(" ", *strings),
        formats_2.format("precision", *precision),
        formats_2.format("recall", *recall),
        formats_2.format("f1", *f1),
        "{:<10}{:>14.2f}".format("acc-all", acc)
    ]
    if console:
        print('\n'.join(contents))
    if file:
        with open(save_dir + "/score.txt") as f:
            f.write('\n'.join(contents))
