import torch
import torch.nn as nn
import numpy as np


def conv_out_shape(in_shape, kernel_count, kernel_size, stride, padding):
    _, n, m = in_shape
    a = (n + 2 * padding - kernel_size) // stride + 1
    b = (m + 2 * padding - kernel_size) // stride + 1
    return (kernel_count, a, b)


def pool_out_shape(in_shape, pool_size, stride, padding):
    c, n, m = in_shape
    a = (n + 2 * padding - pool_size) // stride + 1
    b = (m + 2 * padding - pool_size) // stride + 1
    return (c, a, b)


def make_activation_layer(activation_type='relu'):
    act_type = activation_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    if act_type == 'leaky':
        return nn.LeakyReLU()
    elif act_type == 'sigmoid':
        return nn.Sigmoid()
    elif act_type == 'tanh':
        return nn.Tanh()

    assert False, 'Invalid Activation Layer'


def make_layer(desc_line, in_shape):
    desc = desc_line.strip().split()
    layer_name = desc[0].lower()

    if layer_name == 'conv':
        kernel_count = int(desc[1])
        kernel_size = int(desc[2])
        stride = int(desc[3])
        padding = int(desc[4])
        conv = nn.Conv2d(in_shape[0], kernel_count, kernel_size, stride, padding)
        in_shape = conv_out_shape(in_shape, kernel_count, kernel_size, stride, padding)
        return conv, in_shape

    elif layer_name.startswith('global'):
        in_shape = (in_shape[0], 1, 1)
        if 'max' in layer_name:
            return nn.AdaptiveMaxPool2d(1), in_shape
        elif 'avg' in layer_name:
            return nn.AdaptiveAvgPool2d(1), in_shape

    elif layer_name.endswith('pool'):
        pool_size = int(desc[1])
        stride = int(desc[2])
        padding = int(desc[3])
        in_shape = pool_out_shape(in_shape, pool_size, stride, padding)
        if 'max' in layer_name:
            return nn.MaxPool2d(pool_size, stride, padding), in_shape
        elif 'avg' in layer_name:
            return nn.AvgPool2d(pool_size, stride, padding), in_shape

    elif layer_name == 'flatten':
        return nn.Flatten(), np.prod(in_shape)
    elif layer_name == 'fc':
        output_size = int(desc[1])
        return nn.Linear(in_shape, output_size), output_size
    elif layer_name == 'softmax':
        return nn.Softmax(dim=1), in_shape
    return make_activation_layer(layer_name), in_shape


if __name__ == '__main__':
    x = torch.ones(32, 3, 32, 32)
    conv = nn.Conv2d(3, 6, 5, 1, 0)
    y = conv(x)
    print(y.shape)
    print(conv.weight.shape)
    print(conv.bias.shape)

    glbMaxPool = nn.AdaptiveAvgPool2d(1)
    y = glbMaxPool(y)
    print(y.shape)

    flatten = nn.Flatten()
    y = flatten(y)
    print(y.shape)

    soft = nn.Softmax(dim=1)
    y = soft(y)
    print(y.shape)
    print(y)
