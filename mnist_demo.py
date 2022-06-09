'''
# -*- coding:utf-8 -*-
'''
import argparse
import os
from mindspore import context

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Normal
import mindspore.numpy as mnp
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
# from mindspore.ops import composite as C

# from quantize import *
# 导入模型训练需要的库
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model


parser = argparse.ArgumentParser(description='MindSpore QTTNet Demo')
parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])
args = parser.parse_known_args()[0]
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)


def create_dataset(data_path, batch_size=32, num_parallel_workers=1):
    """定义数据集"""
    mnist_ds = ds.MnistDataset(data_path)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义map映射
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # map映射函数
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # shuffle、batch操作
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)

    # batch_size为每组包含的数据个数（32）。
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)

    return mnist_ds


def tt_conv(in_channels, out_channels, kernel_size, in_channel_mode, out_channel_mode, tt_rank, stride=1, padding=0):
    """
    conv layer TT-core weight initial & Reconstruct original kernel shape
    """
    tt_core_list = []
    tt_core_list.append(initializer(Normal(0.02), [tt_rank[0], kernel_size * kernel_size, tt_rank[1]]))
    for i in range(len(in_channel_mode)):
        tt_core = initializer(Normal(0.02), [tt_rank[i + 1], in_channel_mode[i] * out_channel_mode[i], tt_rank[i + 2]])
        tt_core_list.append(tt_core)
    left_core = mnp.reshape(tt_core_list[0], [-1, tt_rank[1]])
    for i in range(len(tt_core_list)-1):
        right_core = mnp.reshape(tt_core_list[i + 1], [tt_rank[i + 1], -1])
        mul_core = mnp.matmul(left_core, right_core).reshape([-1, tt_rank[i+2]])
        left_core = mul_core

    weight = mnp.reshape(mul_core, [kernel_size] + in_channel_mode + [kernel_size] + out_channel_mode)
    # change modes order
    inch_orders = []
    outch_orders = []
    d = len(in_channel_mode)
    for i in range(d):
        inch_orders.append(1 + i)
        outch_orders.append(2 + d + i)
    weight = mnp.transpose(weight, [0, d + 1] + inch_orders + outch_orders)
    weight = mnp.reshape(weight, [kernel_size, kernel_size, in_channels, out_channels])
    weight = mnp.transpose(weight, (3, 2, 1, 0))

    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


class QTTNet(nn.Cell):
    """
    QTTNet网络结构
    """
    def __init__(self, num_class=10, num_channel=1):
        super(QTTNet, self).__init__()
        # 定义所需要的运算
        self.conv1 = nn.Conv2d(num_channel, 16, 3, pad_mode='valid')
        # TT-Conv
        self.conv2 = tt_conv(16, 64, 3, [2, 4, 2], [4, 4, 4], [1, 32, 32, 32, 1])
        self.fc1 = nn.Dense(2304, 1024, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(1024, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        """使用定义好的运算构建前向网络"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        return x


net = QTTNet()

# 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
print('net loss', net_loss)
# 定义优化器
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)


# 设置模型保存参数
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
# 应用模型保存参数
ckpoint = ModelCheckpoint(prefix="checkpoint_qttnet", config=config_ck)


def train_net(model_train, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """定义训练的方法"""
    # 加载训练数据集
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)

    # dataset_sink_mode用于控制数据时候下沉，数据下沉是指数据通过通道直接传到Device上，
    # 可以加快训练速度，True表示数据下沉。
    model_train.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=sink_mode)


# 通过模型运行测试数据集得到的结果，验证模型的泛化能力。
def test_net(network, model_test, data_path):
    """定义验证的方法"""
    network.eval()
    ds_eval = create_dataset(os.path.join(data_path, "test"))

    # 使用model.eval接口读入测试数据集。使用保存后的模型参数进行推理。
    acc = model_test.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))


# 对数据集进行1个迭代的训练
train_epoch = 1
# 原始数据集路径
mnist_path = "path"
dataset_size = 1
model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
train_net(model, train_epoch, mnist_path, dataset_size, ckpoint, False)
test_net(net, model, mnist_path)
