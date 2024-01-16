import pathlib
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

SCALE = 1
TITLE_SIZE = 14 * SCALE
LABEL_SIZE = 13 * SCALE
TICK_SIZE = 12 * SCALE
ANNOT_SIZE = 8 * SCALE


# 混淆矩阵
def get_confusion_matrix(y_true, y_pred, labels=None):
    """
    计算混淆矩阵
    :param y_true: 真实标签,shape为(N,)
    :param y_pred: 预测标签,shape为(N,),经过分类器后的结果
    :param labels: 标签列表,shape为(n_classes,),默认为None
    :return:  C:array, shape = [n_classes, n_classes],混淆矩阵[i,j]表示真实标签为i,预测标签为j的数量
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


# 绘制混淆矩阵
def plot_confusion_matrix(matrix, normalize=True, save_dir='', names=()) -> None:
    """
    绘制混淆矩阵
    :param matrix: 混淆矩阵,shape为(n_classes,n_classes)
    :param normalize: 是否归一化,默认为True
    :param save_dir: 保存路径
    :param names: 标签列表,shape为(n_classes,),默认为()
    """
    array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True, dpi=1000)
    nc = len(names)  # number of classes, names
    sn.set(font_scale=1.25 if nc < 50 else 0.8)  # for label size
    labels = 0 < nc < 99  # apply names to ticklabels
    ticklabels = names if labels else "auto"
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
        sn.heatmap(array,
                   ax=ax,
                   annot=nc < 30,
                   annot_kws={
                       "size": ANNOT_SIZE},
                   cmap='Blues',
                   fmt='.2f',
                   square=True,
                   vmin=0.0,
                   xticklabels=ticklabels,
                   yticklabels=ticklabels).set_facecolor((1, 1, 1))
    ax.set_xlabel('Predicted', fontdict={'fontsize': LABEL_SIZE}, labelpad=10, rotation=0)
    ax.set_ylabel('True', fontdict={'fontsize': LABEL_SIZE}, labelpad=10, rotation=90)
    ax.set_title('Confusion Matrix', fontsize=TITLE_SIZE, pad=20)
    ax.tick_params(labelsize=TICK_SIZE)
    # time new roman
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontdict={'fontsize': LABEL_SIZE, 'fontname': 'Times New Roman'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontdict={'fontsize': LABEL_SIZE, 'fontname': 'Times New Roman'})
    fig.savefig(Path(save_dir) / 'confusion_matrix.svg', dpi=500)
    plt.close(fig)


def get_score(matrix) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    根据混淆矩阵计算acc, precision,recall,f1
    :param matrix: 混淆矩阵,shape为(n_classes,n_classes)
    :return: acc, precision,recall,f1
    """
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
    precision[precision == 0] = 1e-10  # 防止除0
    f1 = 2 * precision * recall / (precision + recall)
    # 计算acc
    acc = diag.sum() / matrix.sum()
    return acc, precision, recall, f1


def log_scores(scores: tuple, console=False, file=False, save_dir='.', precision=4, nc=10, names=None):
    if not console and not file:  # 不打印到控制台，不保存到文件
        return
    acc, precision_, recall, f1 = scores
    title_fmt = "{:<10}" + "{:>14}" * nc
    score_fmt = "{:<10}" + ("{:>14." + str(precision) + "f}") * nc
    acc_fmt = "{:<10}{:>14." + str(precision) + "f}"
    names = names if names else [f"Label-{i}" for i in range(1, nc + 1)]
    contents = [
        title_fmt.format(" ", *names),
        score_fmt.format("precision", *precision_),
        score_fmt.format("recall", *recall),
        score_fmt.format("f1", *f1),
        acc_fmt.format("acc-all", acc)
    ]
    if console:
        print('\n'.join(contents))
    if file:
        with pathlib.Path(save_dir, "score.txt").open("w") as f:
            f.write('\n'.join(contents))


def log_topk(topk: tuple, k=1, console=False, file=False, save_dir='.', precision=4, nc=10, names=None):
    """
    打印top-k准确率
    :param topk: top_k函数的返回值
    :param k: top-k等级
    :param console: 是否打印到控制台
    :param file: 是否保存到文件
    :param save_dir: 保存目录
    :param precision: 精度
    :param nc: 类别数
    :param names: 类别名称
    :return: None
    """
    if not console and not file:  # 不打印到控制台，不保存到文件
        return
    title_fmt = "{:<10}" + "{:>14}" * nc
    formats_2 = "{:<10}" + ("{:>14." + str(precision) + "f}") * nc
    avg_fmt = "{:<10}{:>14." + str(precision) + "f}"
    names = names if names else [f"Label-{i}" for i in range(1, nc + 1)]
    contents = [
        title_fmt.format(" ", *names),
        formats_2.format(f"top{k}", *topk[1]),
        avg_fmt.format(f"top{k}-all", topk[0]),
    ]
    if console:
        print('\n'.join(contents))
    if file:
        with pathlib.Path(save_dir, "topk.txt").open("w") as f:
            f.write('\n'.join(contents))


def save_matrix(matrix, save_dir, labels=None):
    mt = pd.DataFrame(matrix, index=labels, columns=labels)
    mt.to_excel(save_dir / "confusion_matrix.xlsx")


def tp_fp_tn_fn(matrix, names=None, save_dir="./"):
    """
    计算每个类的tp,fp,tn,fn
    :param matrix: 混淆矩阵,shape为(n_classes,n_classes)
    :param names: 标签列表,shape为(n_classes,),默认为None
    :param save_dir: 保存路径
    :return: tp, fp, tn, fn
    """
    # 计算每个类的tp,fp,tn,fn
    # tp表示预测为正样本且标签为正样本的数量
    # fp表示预测为正样本但标签为负样本的数量
    # tn表示预测为负样本且标签为负样本的数量
    # fn表示预测为负样本但标签为正样本的数量
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    tn = matrix.sum() - (tp + fp + fn)
    pd.DataFrame([tp, fp, tn, fn], index=['tp', 'fp', 'tn', 'fn'], columns=names).to_excel(pathlib.Path(save_dir) / "tp_fp_tn_fn.xlsx")
    return tp, fp, tn, fn


def top_k(y_ture, y_pred, k=1, pre_cls=False):
    """
    计算top-k准确率
    :param pre_cls:  是否计算每个类的top-k准确率
    :param y_ture: 真实标签,shape为(N,)
    :param y_pred: 预测标签,shape为(N,C)
    :param k: top-k
    :return: top-k准确率,总的,None或总的,每个类的
    """
    assert len(y_ture) == len(y_pred)
    top_k = np.argsort(y_pred, axis=1)[:, -k:]  # shape为(N,k)
    y_ture = y_ture.reshape(-1, 1)  # 转为二维数组,shape为(N,1)
    acc = np.sum(np.any(top_k == y_ture, axis=1)) / len(y_ture)
    if pre_cls:
        topk = []
        for i in range(10):
            topk.append(np.sum(top_k[y_ture == i] == i) / len(y_ture[y_ture == i]))
        return acc, np.array(topk)
    return acc, None
