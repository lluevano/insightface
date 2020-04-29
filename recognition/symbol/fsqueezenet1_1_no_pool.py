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
    act = Act(squeeze_1x1, act_type, name=name)
    return act


def fire_module(data, num_filter_squeeze, num_filter_fire, act_type, name, kernel_sequeeze=(1, 1), kernel_1x1=(1, 1),
                kernel_3x3=(3, 3), stride_squeeze=(1, 1), stride_1x1=(1, 1), stride_3x3=(1, 1), pad_1x1=(0, 0),
                pad_3x3=(1, 1), mirror_attr={}):
    name1 = name + "_act_squeeze_1x1"
    squeeze_1x1 = squeeze(data, num_filter_squeeze, name1, kernel_sequeeze, stride_squeeze)
    expand1x1 = mx.symbol.Convolution(data=squeeze_1x1, num_filter=num_filter_fire, kernel=kernel_1x1, stride=stride_1x1, pad=pad_1x1)
    name2 = name + "_act_expand1x1"
    act_expand1x1 = Act(expand1x1, act_type, name=name2)

    expand3x3 = mx.symbol.Convolution(data=squeeze_1x1, num_filter=num_filter_fire, kernel=kernel_3x3,
                                      stride=stride_3x3, pad=pad_3x3)
    name3 = name + "_act_expand3x3"
    act_expand3x3 = Act(expand3x3, act_type, name=name3)
    return act_expand1x1 + act_expand3x3


def get_symbol():
    num_classes = config.emb_size
    act_type = config.net_act
    print('in_network', config)
    data = mx.symbol.Variable(name="data")
    data = data - 127.5
    data = data * 0.0078125

    conv1 = mx.symbol.Convolution(data=data, num_filter=64, kernel=(3, 3), stride=(2, 2), pad=(0, 0))
    act_conv1 = Act(data=conv1, act_type=config.net_act, name="conv_1_activation")
    #pool_conv1 = mx.symbol.Pooling(data=act_conv1, kernel=(3, 3), stride=(2, 2), pool_type='max', attr={})

    fire2 = fire_module(act_conv1, num_filter_squeeze=16, num_filter_fire=64, act_type=config.net_act, name="fire2")
    fire3 = fire_module(fire2, num_filter_squeeze=16, num_filter_fire=64, act_type=config.net_act, name="fire3")

    pool3 = mx.symbol.Pooling(data=fire3, kernel=(3, 3), stride=(2, 2), pool_type='max', attr={})

    fire4 = fire_module(pool3, num_filter_squeeze=32, num_filter_fire=128, act_type=config.net_act, name="fire4")
    fire5 = fire_module(fire4, num_filter_squeeze=32, num_filter_fire=128, act_type=config.net_act, name="fire5")

    pool4 = mx.symbol.Pooling(data=fire5, kernel=(3, 3), stride=(2, 2), pool_type='max', attr={})

    fire6 = fire_module(pool4, num_filter_squeeze=48, num_filter_fire=192, act_type=config.net_act, name="fire6")
    fire7 = fire_module(fire6, num_filter_squeeze=48, num_filter_fire=192, act_type=config.net_act, name="fire7")
    fire8 = fire_module(fire7, num_filter_squeeze=64, num_filter_fire=256, act_type=config.net_act, name="fire8")
    fire9 = fire_module(fire8, num_filter_squeeze=64, num_filter_fire=256, act_type=config.net_act, name="fire9")

    drop9 = mx.sym.Dropout(data=fire9, p=0.5)

    conv10 = mx.symbol.Convolution(data=drop9, num_filter=1000, kernel=(1, 1), stride=(1, 1), pad=(1, 1))
    act_conv10 = Act(data=conv10, act_type=config.net_act, name="conv_10_activation")
    pool10 = mx.sym.Pooling(data=act_conv10, kernel=(13, 13), global_pool=True, pool_type='avg')
	
    fc1 = mx.sym.flatten(data=pool10, name='fc1')
    return fc1
