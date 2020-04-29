import sys
import os
import mxnet as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config
import symbol_utils


def channel_shuffle(data, groups):
    data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
    data = mx.sym.swapaxes(data, 1, 2)
    data = mx.sym.reshape(data, shape=(0, -3, -2))
    return data


def Activation(data, act_type):
    if act_type == 'prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu')
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type)
    return body


def shuffleUnit(residual, in_channels, out_channels, split, act_type):
    # for guideline 1
    equal_channels = out_channels//2

    if split == True:
        DWConv_stride = 1
        # split feature map
        branch1 = mx.sym.slice_axis(residual, axis=1, begin=0, end=in_channels // 2)
        branch2 = mx.sym.slice_axis(residual, axis=1, begin=in_channels // 2, end=in_channels)
    else:
        DWConv_stride = 2
        branch1 = residual
        branch2 = residual

        branch1 = mx.sym.Convolution(data=branch1, num_filter=in_channels, kernel=(3, 3), pad=(1, 1),
                                     stride=(DWConv_stride, DWConv_stride), num_group=in_channels, no_bias=1)
        branch1 = mx.sym.BatchNorm(data=branch1)

        branch1 = mx.sym.Convolution(data=branch1, num_filter=equal_channels, kernel=(1, 1), stride=(1, 1), no_bias=1)
        branch1 = mx.sym.BatchNorm(data=branch1)
        branch1 = Activation(data=branch1, act_type=act_type)

    branch2 = mx.sym.Convolution(data=branch2, num_filter=equal_channels, kernel=(1, 1), stride=(1, 1), no_bias=1)
    branch2 = mx.sym.BatchNorm(data=branch2)
    branch2 = Activation(data=branch2, act_type=act_type)

    branch2 = mx.sym.Convolution(data=branch2, num_filter=equal_channels, kernel=(3, 3), pad=(1, 1),
                                 stride=(DWConv_stride, DWConv_stride), num_group=equal_channels, no_bias=1)
    branch2 = mx.sym.BatchNorm(data=branch2)

    branch2 = mx.sym.Convolution(data=branch2, num_filter=equal_channels, kernel=(1, 1), stride=(1, 1), no_bias=1)
    branch2 = mx.sym.BatchNorm(data=branch2)
    branch2 = Activation(data=branch2, act_type=act_type)

    data = mx.sym.concat(branch1, branch2, dim=1)
    data = channel_shuffle(data=data, groups=2)

    return data


def make_stage(data, stage, multiplier=1, act_type='prelu'):
    stage_repeats = [3, 7, 3]

    if multiplier == 0.5:
        out_channels = [-1, 24, 48, 96, 192]
    elif multiplier == 1:
        out_channels = [-1, 24, 116, 232, 464]
    elif multiplier == 1.5:
        out_channels = [-1, 24, 176, 352, 704]
    elif multiplier == 2:
        out_channels = [-1, 24, 244, 488, 976]

    # DWConv_stride = 2
    data = shuffleUnit(data, out_channels[stage - 1], out_channels[stage], split=False, act_type=act_type)
    # DWConv_stride = 1
    for i in range(stage_repeats[stage - 2]):
        data = shuffleUnit(data, out_channels[stage], out_channels[stage], split=True, act_type=act_type)

    return data


def get_symbol():
    depth_multiplier = config.depth_multiplier
    act_type = config.net_act

    num_classes = config.emb_size
    print('in_network', config)
    fc_type = config.net_output
    data = mx.symbol.Variable(name="data")
    print("len_data: "+str(len(data)))
    data = data - 127.5
    data = data * 0.0078125

    data = mx.sym.Convolution(data=data, num_filter=24, kernel=(3, 3), stride=(2, 2), pad=(1, 1), no_bias=1)
    data = mx.sym.BatchNorm(data=data)
    data = Activation(data=data, act_type=act_type)

    data = make_stage(data, 2, depth_multiplier, act_type=act_type)

    data = make_stage(data, 3, depth_multiplier, act_type=act_type)

    data = make_stage(data, 4, depth_multiplier, act_type=act_type)

    # extra_conv
    final_channels = 1024 if depth_multiplier != '2.0' else 2048
    extra_conv = mx.sym.Convolution(data=data, num_filter=final_channels,
                                    kernel=(1, 1), stride=(1, 1), no_bias=1)
    extra_conv = mx.sym.BatchNorm(data=extra_conv)
    data = Activation(data=extra_conv, act_type=act_type)

    fc1 = symbol_utils.get_fc1(data, num_classes, fc_type)
    return fc1
