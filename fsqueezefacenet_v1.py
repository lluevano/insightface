#conding:utf-8
import sys
import os
import mxnet as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config
import symbol_utils


def Act(data, act_type, name):
    # ignore param act_type, set it in this function
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.sym.Activation(data=data, act_type=act_type, name=name)
    return body


def squeeze(data, num_filter, name, kernel=(1, 1), stride=(1, 1), pad=(0, 0), act_type="prelu", mirror_attr={}):
    squeeze_1x1 = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    # act = mx.symbol.Activation(data = squeeze_1x1, act_type=act_type, attr=mirror_attr)
    act = Act(squeeze_1x1, act_type, name=name)
    return act


def fire_module(data, num_filter_squeeze, num_filter_fire, act_type, name, kernel_sequeeze=(1, 1), kernel_1x1=(1, 1),
                kernel_3x3=(3, 3), stride_squeeze=(1, 1), stride_1x1=(1, 1), stride_3x3=(1, 1), pad_1x1=(0, 0),
                pad_3x3=(1, 1), mirror_attr={}):
    name1 = name + "_act_squeeze_1x1"
    squeeze_1x1 = squeeze(data, num_filter_squeeze, name1, kernel_sequeeze, stride_squeeze)
    expand1x1 = mx.symbol.Convolution(data=squeeze_1x1, num_filter=num_filter_fire, kernel=kernel_1x1,
                                      stride=stride_1x1, pad=pad_1x1)
    name2 = name + "_act_expand1x1"
    act_expand1x1 = Act(expand1x1, act_type, name=name2)
    # relu_expand1x1 = mx.symbol.Activation(data=expand1x1, act_type=act_type, attr=mirror_attr)

    expand3x3 = mx.symbol.Convolution(data=squeeze_1x1, num_filter=num_filter_fire, kernel=kernel_3x3,
                                      stride=stride_3x3, pad=pad_3x3)
    # relu_expand3x3 = mx.symbol.Activation(data=expand3x3, act_type=act_type, attr=mirror_attr)
    name3 = name + "_act_expand3x3"
    act_expand3x3 = Act(expand3x3, act_type, name=name3)
    return act_expand1x1+act_expand3x3


def get_symbol():
    num_classes = config.emb_size
    print('in_network', config)
    fc_type = config.net_output
    data = mx.symbol.Variable(name="data")
    data = data - 127.5
    data = data * 0.0078125

    conv1 = mx.symbol.Convolution(data=data, num_filter=96, kernel=(7, 7), stride=(2, 2), pad=(0, 0))
    act_conv1 = Act(data=conv1, act_type=config.net_act, name="conv_1_activation")
    pool_conv1 = mx.symbol.Pooling(data=act_conv1, kernel=(3, 3), stride=(2, 2), pool_type='max', attr={})

    fire2 = fire_module(pool_conv1, num_filter_squeeze=16, num_filter_fire=64, act_type=config.net_act, name="fire2")
    fire3 = fire_module(fire2, num_filter_squeeze=16, num_filter_fire=64, act_type=config.net_act, name="fire3")
    fire4 = fire_module(fire3, num_filter_squeeze=32, num_filter_fire=128, act_type=config.net_act, name="fire4")

    pool4 = mx.symbol.Pooling(data=fire4, kernel=(3, 3), stride=(2, 2), pool_type='max', attr={})

    fire5 = fire_module(pool4, num_filter_squeeze=32, num_filter_fire=128, act_type=config.net_act, name="fire5")
    fire6 = fire_module(fire5, num_filter_squeeze=48, num_filter_fire=192, act_type=config.net_act, name="fire6")
    fire7 = fire_module(fire6, num_filter_squeeze=48, num_filter_fire=192, act_type=config.net_act, name="fire7")
    fire8 = fire_module(fire7, num_filter_squeeze=64, num_filter_fire=256, act_type=config.net_act, name="fire8")

    pool8 = mx.symbol.Pooling(data=fire8, kernel=(3, 3), stride=(2, 2), pool_type='max', attr={})

    fire9 = fire_module(pool8, num_filter_squeeze=64, num_filter_fire=256, act_type=config.net_act, name="fire9")
    drop9 = mx.sym.Dropout(data=fire9, p=0.5)

    conv10 = mx.symbol.Convolution(data=drop9, num_filter=512, kernel=(1, 1), stride=(1, 1), pad=(1, 1))
    act_conv10 = Act(data=conv10, act_type=config.net_act, name="conv_10_activation")
    # pool10 = mx.symbol.Pooling(data=act_conv10, kernel=(13, 13), pool_type='avg', attr={})
    #
    # flatten = mx.symbol.Flatten(data=pool10, name='flatten')
    # softmax = mx.symbol.SoftmaxOutput(data=flatten, name='softmax')

    fc1 = symbol_utils.get_fc1(act_conv10, num_classes, fc_type)
    return fc1
