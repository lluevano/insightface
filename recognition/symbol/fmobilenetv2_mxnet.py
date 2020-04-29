import sys
import os
import mxnet as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config
import symbol_utils


def Activation(data, act_type):
    if act_type == 'prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu')
    elif act_type == 'relu6':
      body = mx.sym.clip(data, 0, 6)
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type)
    return body

	
def ConvBlock(data, num_filter, kernel_size, strides, act_type, name):
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel_size, stride=strides, pad=(1, 1), no_bias=1)   
    data = mx.sym.BatchNorm(data=data, fix_gamma=True)
    data = Activation(data=data, act_type=act_type)   
    return data	

	
def Conv1x1(data, num_filter, act_type, is_linear=False):
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), pad=(0, 0), no_bias=1)   
    data = mx.sym.BatchNorm(data=data, fix_gamma=True)

    if not is_linear:
        data = Activation(data=data, act_type=act_type)
    return data	

	
def DWise(data, num_filter, stride, act_type):
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1, 1), num_group=num_filter, no_bias=1)     
    data = mx.sym.BatchNorm(data=data, fix_gamma=True)
    data = Activation(data=data, act_type=act_type)    
    return data


def Bottleneck(data, exp_factor, input_size, num_filter, stride, act_type, same_shape=True):
    data = Conv1x1(data=data, num_filter=exp_factor*input_size, act_type=act_type)
    data = DWise(data=data, num_filter=exp_factor*input_size, stride=stride, act_type=act_type)
    data = Conv1x1(data=data, num_filter=num_filter, act_type=act_type, is_linear=True)

    if stride == 1 and not same_shape:
         residuals =  Conv1x1(data=data, num_filter=num_filter, act_type=act_type)
         data = data + residuals

    return data


def get_symbol():
    width_multiplier = config.width_multiplier
    act_type = config.net_act

    num_classes = config.emb_size
    print('in_network', config)
    fc_type = config.net_output
    data = mx.symbol.Variable(name="data")
    data = data - 127.5
    data = data * 0.0078125

    cn = [int(x*width_multiplier) for x in [32, 16, 24, 32, 64, 96, 160, 320]]

    if config.net_input==0:
      conv = ConvBlock(data, num_filter=cn[0], kernel_size=(3, 3), strides=(2, 2), act_type=act_type, name="conv1") 
    else:
      conv = ConvBlock(data, num_filter=cn[0], kernel_size=(3, 3), strides=(1, 1), act_type=act_type, name="conv1") 
    
    data = Bottleneck(conv, exp_factor=1, input_size=cn[0], num_filter=cn[1], stride=(1, 1), act_type=act_type, same_shape=False)
    data = Bottleneck(data, exp_factor=6, input_size=cn[1], num_filter=cn[2], stride=(2, 2), act_type=act_type, same_shape=False)
    data = Bottleneck(data, exp_factor=6, input_size=cn[1], num_filter=cn[2], stride=(1, 1), act_type=act_type)
    data = Bottleneck(data, exp_factor=6, input_size=cn[2], num_filter=cn[3], stride=(2, 2), act_type=act_type, same_shape=False)
    data = Bottleneck(data, exp_factor=6, input_size=cn[2], num_filter=cn[3], stride=(1, 1), act_type=act_type)
    data = Bottleneck(data, exp_factor=6, input_size=cn[2], num_filter=cn[3], stride=(1, 1), act_type=act_type)
    data = Bottleneck(data, exp_factor=6, input_size=cn[3], num_filter=cn[4], stride=(2, 2), act_type=act_type, same_shape=False)
    data = Bottleneck(data, exp_factor=6, input_size=cn[3], num_filter=cn[4], stride=(1, 1), act_type=act_type)
    data = Bottleneck(data, exp_factor=6, input_size=cn[3], num_filter=cn[4], stride=(1, 1), act_type=act_type)
    data = Bottleneck(data, exp_factor=6, input_size=cn[3], num_filter=cn[4], stride=(1, 1), act_type=act_type)
    data = Bottleneck(data, exp_factor=6, input_size=cn[4], num_filter=cn[5], stride=(1, 1), act_type=act_type, same_shape=False)
    data = Bottleneck(data, exp_factor=6, input_size=cn[4], num_filter=cn[5], stride=(1, 1), act_type=act_type)
    data = Bottleneck(data, exp_factor=6, input_size=cn[4], num_filter=cn[5], stride=(1, 1), act_type=act_type)
    data = Bottleneck(data, exp_factor=6, input_size=cn[5], num_filter=cn[6], stride=(2, 2), act_type=act_type, same_shape=False)
    data = Bottleneck(data, exp_factor=6, input_size=cn[5], num_filter=cn[6], stride=(1, 1), act_type=act_type)
    data = Bottleneck(data, exp_factor=6, input_size=cn[5], num_filter=cn[6], stride=(1, 1), act_type=act_type)
    data = Bottleneck(data, exp_factor=6, input_size=cn[6], num_filter=cn[7], stride=(1, 1), act_type=act_type, same_shape=False)

    if width_multiplier > 1.0:
        last_channels = int(1280*width_multiplier)
    else:
        last_channels = 1280
    conv2 = Conv1x1(data, num_filter=last_channels, act_type=act_type)

    data = mx.sym.Pooling(data=conv2, kernel=(7, 7), global_pool=True, pool_type='avg')
    fc1 = mx.sym.flatten(data=data, name='fc1')

    return fc1
