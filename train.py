import argparse
import random
import time

from matplotlib import pyplot as plt
from model import NeuralNetwork
from utils import increment_path, get_root, LOGGER, colorstr, time_format
from utils.confusion_matrix import log_scores, log_topk
from utils.data import MNIST, save_data, show_save_fig
import val as validate
from tqdm import tqdm

ROOT = get_root(__file__, 0)


def main(opt):
    mnist = MNIST(root=opt.data, transform=[lambda x: x / 255.0,
                                            lambda x: x.reshape(x.shape[0], -1)])  # transform 原始数据 范围0-255,shape 28x28
    # 数据集
    train_images, train_labels = mnist.train_images, mnist.train_labels
    save_dir = increment_path(opt.project / opt.name)
    save_dir.mkdir(parents=True, exist_ok=True)
    # 可视化,随机显示16张图片
    if opt.show:
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            img_idx = random.randint(0, len(train_images))
            plt.imshow(train_images[img_idx].reshape(28, 28), cmap="gray")
            plt.title(f"Label: {train_labels[img_idx]}")
            plt.axis("off")
        plt.show()
    # 超参数
    lr = opt.lr
    epochs = opt.epochs
    batch_size = opt.batch_size
    hidden_layer_num = opt.hidden_layer_num
    hidden_size = opt.hidden_size

    net = NeuralNetwork(in_size=784, out_size=10, hidden_layer_num=hidden_layer_num, hidden_size=hidden_size)

    train_loss_list = []
    test_loss_list = []
    accuracy_list = []
    main_batch = len(train_images) // batch_size
    res_batch = len(train_images) % batch_size
    bar_total = main_batch + (1 if res_batch else 0)
    t1 = time.time()
    for epoch in range(epochs):
        net.train()
        batch_index = 0
        mt_loss = 0.0  # 平均训练损失，累计移动平均
        pbar = tqdm(range(bar_total), bar_format="{l_bar}{bar:10}{r_bar}")
        while batch_index < len(train_images):  # 小批量梯度下降
            pbar.set_description(f"[Epoch {epoch + 1}/{epochs}]")
            batch_x = train_images[batch_index: batch_index + batch_size]
            batch_y = train_labels[batch_index: batch_index + batch_size]
            batch_index += batch_size
            # forward
            y_pred = net(batch_x)
            # backward
            net.backward(batch_y, y_pred)
            # update
            net.update(lr)
            # loss
            loss = net.loss(y_pred, batch_y)
            mt_loss = (mt_loss * batch_index + loss) / (batch_index + 1)
            pbar.set_postfix(loss=mt_loss)
            pbar.update()
        pbar.close()
        # 验证
        (acc, precision, recall, f1), mv_loss, (mtop1,top1s) = validate.run(data=mnist, net=net)
        # 记录
        train_loss_list.append(mt_loss)
        test_loss_list.append(mv_loss)
        accuracy_list.append(acc)
        LOGGER.info(colorstr('green', f"Train Loss: {mt_loss:.4f}, Test Loss: {mv_loss:.4f}, Accuracy: {acc:.4f} Top1: {mtop1:.4f} \n"))
        # log_scores((acc, precision, recall, f1), console=True, file=False)
        # log_topk((mtop1,top1s), console=True, file=False)
    t2 = time.time()
    LOGGER.info(colorstr("Train Done! Finished with epochs " + colorstr('red', f"{epochs}") + colorstr(" in ") + colorstr('red', f"{time_format(t2 - t1)}")))
    net.save(f"{save_dir}/params.pkl")
    LOGGER.info(f"Save model to {save_dir.as_posix()}/params.pkl")
    # 保存数据
    save_data(train_loss_list, f"{save_dir}/train_loss.txt")
    save_data(test_loss_list, f"{save_dir}/test_loss.txt")
    save_data(accuracy_list, f"{save_dir}/accuracy.txt")
    # 可视化并保存
    show_save_fig(train_loss_list, y_label="loss", title="Train Loss", save_path=f"{save_dir}/train_loss.png")
    show_save_fig(test_loss_list, y_label="loss", title="Test Loss", save_path=f"{save_dir}/test_loss.png")
    show_save_fig(accuracy_list, y_label="accuracy", title="Accuracy", save_path=f"{save_dir}/accuracy.png")


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--hidden-layer-num", type=int, default=3, help="hidden layer num")
    parser.add_argument("--hidden-size", type=int, nargs="+", default=[256, 128, 64], help="hidden size")
    parser.add_argument("--project", type=str, default=ROOT / "logs", help="save path")
    parser.add_argument("--name", type=str, default="train", help="save name")
    parser.add_argument("--data", type=str, default=ROOT / "data", help="data root")
    parser.add_argument("--show", action="store_true", help="show image")
    args = parser.parse_args()
    return args


def run(**kwargs):
    opt = parser_opt()
    opt.__dict__.update(kwargs)
    main(opt)
    return opt


if __name__ == '__main__':
    opt = parser_opt()
    main(opt)
