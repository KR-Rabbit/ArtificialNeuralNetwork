import numpy as np


def test(net, test_images, test_labels, batch_size=64):
    batch_index = 0
    correct = 0  # 正确数
    total = 0  # 总数
    loss = 0  # 损失
    while batch_index < len(test_images):
        batch_x = test_images[batch_index: batch_index + batch_size]
        batch_y = test_labels[batch_index: batch_index + batch_size]
        batch_index += batch_size
        y_pred = np.argmax(net.forward(batch_x), axis=1)
        loss += net.loss(batch_x, batch_y)
        total += len(batch_y)
        correct += (y_pred == batch_y).sum().item()
    return loss / total, correct / total,
