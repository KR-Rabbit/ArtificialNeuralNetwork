import numpy as np


def validate(net, test_images, test_labels, batch_size=64):
    batch_index = 0
    correct = 0
    total = 0
    while batch_index < len(test_images):
        batch_x = test_images[batch_index: batch_index + batch_size]
        batch_y = test_labels[batch_index: batch_index + batch_size]
        batch_index += batch_size
        y_pred = np.argmax(net.forward(batch_x), axis=1, keepdims=True)
        total += batch_y.size(0)
        correct += (y_pred == batch_y).sum().item()
    return correct / total
