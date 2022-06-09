'''
# -*- coding:utf-8 -*-
'''
import mindspore as ms
import mindspore.numpy as mnp
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
import numpy as np
bitsw = 8
bitsa = 8


def scale(x):
    """scale"""
    scale_ = ms.Tensor.max(ms.Tensor.abs(x))
    result = 2. ** mnp.round(mnp.log2(scale_))
    return result


def delta(bits):
    """delta"""
    result = (2. ** (1 - bits))
    return result


def clip(x, bits):
    """clip"""
    if bits >= 32:
        step = 0
    else:
        step = delta(bits)
    ceil = 1 - step
    floor = step - 1
    result = np.clamp(x, floor, ceil)
    return result


def quant(x, bits):
    """quant"""
    if bits >= 32:
        result = x
    else:
        result = mnp.round(x / delta(bits)) * delta(bits)
    return result


def qw(x):
    """qw"""
    bits = bitsw
    if bits >= 32:
        result = x
    else:
        result = np.clip(quant(x, bits), bits)
    return result


def qa(x):
    """qa"""
    bits = bitsa
    if bits >= 32:
        result = x
    else:
        result = quant(x, bits)
    return result


class QW(PrimitiveWithInfer):
    """
    量化部分（自定义梯度计算）包含了自定义算子与原语注册，会在后续实现
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def construct(self, x):
        """construct"""
        result = qw(x)
        return result

    def bprop(self, grad_output):
        """bprop"""
        grad_input = grad_output
        return grad_input


quantizew = QW()


class QA(PrimitiveWithInfer):
    """
    量化部分（自定义梯度计算）包含了自定义算子与原语注册，会在后续实现
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def construct(self, x):
        """construct"""
        self.save_for_backward(x)
        result = qa(x)
        return result

    def bprop(self, grad_output):
        """bprop"""
        grad_input = grad_output
        return grad_input


quantize_ae = QA.apply
