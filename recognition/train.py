from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import sklearn
import pickle
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
from config import config, default, generate_config
from metric import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import flops_counter
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
import verification
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbol'))
import fresnet
import fmobilefacenet
import fmobilenet
import fmnasnet
import fdensenet
#custom network import
import fsqueezefacenet_v1
import fsqueezefacenet_v2
import fshufflefacenetv2
import fefficientnet
import fshufflenetv1
import fsqueezenet1_0
import fsqueezenet1_1
import fsqueezenet1_2
import fmobilenetv2
import fsqueezenet1_1_no_pool
import fmobilefacenetv1
import fmobilenetv2_mxnet
import vargfacenet
import mobilenetv3

import sys
if sys.version_info < (3,):
    range = xrange

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler('current_model.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)
#logging.basicConfig(filename='current_model_output.txt',level=logging.INFO) 
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)
#logging.basicConfig(filename='current_model_output.txt',level=logging.INFO) 

args = None



def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--dataset', default=default.dataset, help='dataset config')
  parser.add_argument('--network', default=default.network, help='network config')
  parser.add_argument('--loss', default=default.loss, help='loss config')
  args, rest = parser.parse_known_args()
  generate_config(args.network, args.dataset, args.loss)
  parser.add_argument('--models-root', default=default.models_root, help='root directory to save model.')
  parser.add_argument('--pretrained', default=default.pretrained, help='pretrained model to load')
  parser.add_argument('--pretrained-epoch', type=int, default=default.pretrained_epoch, help='pretrained epoch to load')
  parser.add_argument('--ckpt', type=int, default=default.ckpt, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--verbose', type=int, default=default.verbose, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--lr', type=float, default=default.lr, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default=default.lr_steps, help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=default.wd, help='weight decay')
  parser.add_argument('--mom', type=float, default=default.mom, help='momentum')
  parser.add_argument('--frequent', type=int, default=default.frequent, help='')
  parser.add_argument('--per-batch-size', type=int, default=default.per_batch_size, help='batch size in each context')
  parser.add_argument('--kvstore', type=str, default=default.kvstore, help='kvstore setting')
  args = parser.parse_args()
  return args


def get_symbol(args):
  embedding = eval(config.net_name).get_symbol() #obtengo la red
  all_label = mx.symbol.Variable('softmax_label') #creo la variable softmax_label
  gt_label = all_label #asigno el apuntador a gt_label
  is_softmax = True 
  _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size), 
    lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
  if config.loss_name=='softmax': #reviso si esta llamando a softmax
    if config.fc7_no_bias:
      fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
    else:
      _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
      fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=config.num_classes, name='fc7')
  elif config.loss_name=='margin_softmax': #esta es la configuracion de arcface, cosface y combined, solo cambian los parametros del config
    s = config.loss_s
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    #normaliza
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
    #constantes de loss functions declaradas en config.py
    #is_nsoftmax = (config.loss_m1 == 1) and (config.loss_m2 == 0) and (config.loss_m3==0)
    #is_arcface = (config.loss_m1 == 1) and (config.loss_m2 == 0.5) and (config.loss_m3==0)
    #is_cosface = (config.loss_m1 == 1) and (config.loss_m2 == 0) and (config.loss_m3==0.35)
    #is_combined = (config.loss_m1 == 1) and (config.loss_m2 == .3) and (config.loss_m3==0.2)
    
    if config.loss_m1!=1.0 or config.loss_m2!=0.0 or config.loss_m3!=0.0: #not any of the above or arcface/combined or cosface/combined
      if config.loss_m1==1.0 and config.loss_m2==0.0: #is_cosface
        s_m = s*config.loss_m3
        gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = s_m, off_value = 0.0)
        fc7 = fc7-gt_one_hot
      else: #is arcface or combined
        zy = mx.sym.pick(fc7, gt_label, axis=1)
        cos_t = zy/s
        t = mx.sym.arccos(cos_t)
        if config.loss_m1!=1.0:
          t = t*config.loss_m1
        if config.loss_m2>0.0:
          t = t+config.loss_m2
        body = mx.sym.cos(t)
        if config.loss_m3>0.0:
          body = body - config.loss_m3
        new_zy = body*s
        diff = new_zy - zy
        diff = mx.sym.expand_dims(diff, 1)
        gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 1.0, off_value = 0.0)
        body = mx.sym.broadcast_mul(gt_one_hot, diff)
        fc7 = fc7+body
        
  out_list = [mx.symbol.BlockGrad(embedding)]
  if is_softmax:
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid')
    out_list.append(softmax)
    if config.ce_loss:
      body = mx.symbol.SoftmaxActivation(data=fc7)
      body = mx.symbol.log(body)
      _label = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = -1.0, off_value = 0.0)
      body = body*_label
      ce_loss = mx.symbol.sum(body)/args.per_batch_size
      out_list.append(mx.symbol.BlockGrad(ce_loss))
  else: #any of the other cases
    out_list.append(mx.sym.BlockGrad(gt_label))
  out = mx.symbol.Group(out_list)
  return out

def train_net(args):
    ctx = []
    #ctx.append(mx.gpu(1)) #manual
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
      for i in range(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      logging.info('use cpu')
    else:
      logging.info('gpu num: %d', len(ctx))
    prefix = os.path.join(args.models_root, '%s-%s-%s'%(args.network, args.loss, args.dataset), 'model')
    prefix_dir = os.path.dirname(prefix)
    logging.info('prefix %s', prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    args.ctx_num = len(ctx)
    args.batch_size = args.per_batch_size*args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = config.image_shape[2]
    config.batch_size = args.batch_size
    config.per_batch_size = args.per_batch_size

    data_dir = config.dataset_path
    path_imgrec = None
    path_imglist = None
    image_size = config.image_shape[0:2]
    assert len(image_size)==2
    assert image_size[0]==image_size[1]
    logging.info('image_size %s', str(image_size))
    logging.info('num_classes %d', config.num_classes)
    path_imgrec = os.path.join(data_dir, "train.rec")

    logging.info('Called with argument: %s %s', str(args), str(config))
    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None

    begin_epoch = args.pretrained_epoch
    if len(args.pretrained)==0: #no pretraining
      arg_params = None
      aux_params = None
      sym = get_symbol(args)
      if config.net_name=='spherenet':
        data_shape_dict = {'data' : (args.per_batch_size,)+data_shape}
        spherenet.init_weights(sym, data_shape_dict, args.num_layers)
    else: #load pretrained model
      logging.info('loading %s %s', str(args.pretrained), str(args.pretrained_epoch))
      _, arg_params, aux_params = mx.model.load_checkpoint(args.pretrained, args.pretrained_epoch)
      sym = get_symbol(args)

    if config.count_flops:
      all_layers = sym.get_internals()
      _sym = all_layers['fc1_output']
      FLOPs = flops_counter.count_flops(_sym, data=(1,3,image_size[0],image_size[1]))
      _str = flops_counter.flops_str(FLOPs)
      logging.info('Network FLOPs: %s'%_str)

    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)
    model = mx.mod.Module( #executable options and full model is loaded, loss functions and all
        context       = ctx,
        symbol        = sym,
        #fixed_param_names = ["conv_1_conv2d_weight","res_2_block0_conv_sep_conv2d_weight","res_2_block0_conv_dw_conv2d_weight","res_2_block0_conv_proj_conv2d_weight","res_2_block1_conv_sep_conv2d_weight","res_2_block1_conv_dw_conv2d_weight","res_2_block1_conv_proj_conv2d_weight","dconv_23_conv_sep_conv2d_weight","dconv_23_conv_dw_conv2d_weight","dconv_23_conv_proj_conv2d_weight","res_3_block0_conv_sep_conv2d_weight","res_3_block0_conv_dw_conv2d_weight","res_3_block0_conv_proj_conv2d_weight","res_3_block1_conv_sep_conv2d_weight","res_3_block1_conv_dw_conv2d_weight","res_3_block1_conv_proj_conv2d_weight","res_3_block2_conv_sep_conv2d_weight","res_3_block2_conv_dw_conv2d_weight","res_3_block2_conv_proj_conv2d_weight","res_3_block3_conv_sep_conv2d_weight","res_3_block3_conv_dw_conv2d_weight","res_3_block3_conv_proj_conv2d_weight","res_3_block4_conv_sep_conv2d_weight","res_3_block4_conv_dw_conv2d_weight","res_3_block4_conv_proj_conv2d_weight","res_3_block5_conv_sep_conv2d_weight","res_3_block5_conv_dw_conv2d_weight","res_3_block5_conv_proj_conv2d_weight","res_3_block6_conv_sep_conv2d_weight","res_3_block6_conv_dw_conv2d_weight","res_3_block6_conv_proj_conv2d_weight","res_3_block7_conv_sep_conv2d_weight","res_3_block7_conv_dw_conv2d_weight","res_3_block7_conv_proj_conv2d_weight","dconv_34_conv_sep_conv2d_weight","dconv_34_conv_dw_conv2d_weight","dconv_34_conv_proj_conv2d_weight","res_4_block0_conv_sep_conv2d_weight","res_4_block0_conv_dw_conv2d_weight","res_4_block0_conv_proj_conv2d_weight","res_4_block1_conv_sep_conv2d_weight","res_4_block1_conv_dw_conv2d_weight","res_4_block1_conv_proj_conv2d_weight","res_4_block2_conv_sep_conv2d_weight","res_4_block2_conv_dw_conv2d_weight","res_4_block2_conv_proj_conv2d_weight","res_4_block3_conv_sep_conv2d_weight","res_4_block3_conv_dw_conv2d_weight","res_4_block3_conv_proj_conv2d_weight","res_4_block4_conv_sep_conv2d_weight","res_4_block4_conv_dw_conv2d_weight","res_4_block4_conv_proj_conv2d_weight","res_4_block5_conv_sep_conv2d_weight","res_4_block5_conv_dw_conv2d_weight","res_4_block5_conv_proj_conv2d_weight","res_4_block6_conv_sep_conv2d_weight","res_4_block6_conv_dw_conv2d_weight","res_4_block6_conv_proj_conv2d_weight","res_4_block7_conv_sep_conv2d_weight","res_4_block7_conv_dw_conv2d_weight","res_4_block7_conv_proj_conv2d_weight","res_4_block8_conv_sep_conv2d_weight","res_4_block8_conv_dw_conv2d_weight","res_4_block8_conv_proj_conv2d_weight","res_4_block9_conv_sep_conv2d_weight","res_4_block9_conv_dw_conv2d_weight","res_4_block9_conv_proj_conv2d_weight","res_4_block10_conv_sep_conv2d_weight","res_4_block10_conv_dw_conv2d_weight","res_4_block10_conv_proj_conv2d_weight","res_4_block11_conv_sep_conv2d_weight","res_4_block11_conv_dw_conv2d_weight","res_4_block11_conv_proj_conv2d_weight","res_4_block12_conv_sep_conv2d_weight","res_4_block12_conv_dw_conv2d_weight","res_4_block12_conv_proj_conv2d_weight","res_4_block13_conv_sep_conv2d_weight","res_4_block13_conv_dw_conv2d_weight","res_4_block13_conv_proj_conv2d_weight","res_4_block14_conv_sep_conv2d_weight","res_4_block14_conv_dw_conv2d_weight","res_4_block14_conv_proj_conv2d_weight","res_4_block15_conv_sep_conv2d_weight","res_4_block15_conv_dw_conv2d_weight","res_4_block15_conv_proj_conv2d_weight","dconv_45_conv_sep_conv2d_weight","dconv_45_conv_dw_conv2d_weight","dconv_45_conv_proj_conv2d_weight"],
        #fixed_param_names = ['convolution'+str(i)+'_weight' for i in range(1,40)],
    )
    val_dataiter = None

    if config.loss_name.find('triplet')>=0: #if triplet or atriplet loss
      from triplet_image_iter import FaceImageIter
      triplet_params = [config.triplet_bag_size, config.triplet_alpha, config.triplet_max_ap]
      train_dataiter = FaceImageIter(
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = path_imgrec,
          shuffle              = True,
          rand_mirror          = config.data_rand_mirror,
          mean                 = mean,
          cutoff               = config.data_cutoff,
          ctx_num              = args.ctx_num,
          images_per_identity  = config.images_per_identity,
          triplet_params       = triplet_params,
          mx_model             = model,
      )
      _metric = LossValueMetric()
      eval_metrics = [mx.metric.create(_metric)]
    else:
      from image_iter import FaceImageIter
      train_dataiter = FaceImageIter(
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = path_imgrec,
          shuffle              = True,
          rand_mirror          = config.data_rand_mirror,
          mean                 = mean,
          cutoff               = config.data_cutoff,
          color_jittering      = config.data_color,
          images_filter        = config.data_images_filter,
      )
      metric1 = AccMetric()
      eval_metrics = [mx.metric.create(metric1)]
      if config.ce_loss:
        metric2 = LossValueMetric()
        eval_metrics.append( mx.metric.create(metric2) )
    gaussian_nets = ['fresnet','fmobilefacenet','fsqueezefacenet_v1','fsqueezefacenet_v2', 'fshufflefacenetv2', 'fshufflenetv1','fsqueezenet1_0','fsqueezenet1_1', 'fsqueezenet1_2', 'fsqueezenet1_1_no_pool','fmobilenetv2','fmobilefacenetv1','fmobilenetv2_mxnet','vargfacenet', 'mobilenetv3']
    if config.net_name in gaussian_nets:

#    if config.net_name=='fresnet' or config.net_name=='fmobilefacenet' or config.net_name=='fsqueezefacenet_v1' or config.net_name=='fsqueezefacenet_v2' or config.net_name=='fshufflefacenetv2' or config.net_name =='fefficientnet' or config.net_name == 'fshufflenetv1' or config.net_name == 'fsqueezenet1_0' or config.net_name == 'fsqueezenet1_1' or config.net_name =='fsqueezenet1_2' or config.net_name == 'fsqueezenet1_1_no_pool' or config.net_name == 'fmobilenetv2':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
      print("GAUSSIAN INITIALIZER")
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
      print("UNIFORM INITIALIZER")
    #initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    _rescale = 1.0/args.ctx_num
    opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=_rescale)
    #opt = optimizer.Adam(learning_rate=args.lr,wd=args.wd,rescale_grad=_rescale)    
    _cb = mx.callback.Speedometer(args.batch_size, args.frequent)

    ver_list = []
    ver_name_list = []
    for name in config.val_targets:
      path = os.path.join(data_dir,name+".bin")
      if os.path.exists(path):
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        logging.info('ver %s', name)



    def ver_test(nbatch):
      results = []
      for i in range(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10, None, None)
        logging.info('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
        logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
      return results



    highest_acc = [0.0, 0.0]  #lfw and target
    highest_train_acc = [0.0,0.0]
    #for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    logging.info('lr_steps %s', str(lr_steps))
    def _batch_callback(param):
      #global global_step
      #weights_db = model.get_params()[0]['fire2_act_squeeze_1x1'].asnumpy()

      
      
      #weights_db = model.get_params()[0]
      #print(str(weights_db.keys()))

      #for k, v in weights_db.items():
          #print(k)
     #     if(np.any(np.isnan(v.asnumpy())) or np.any(np.isinf(v.asnumpy()))):
     #         print("nan or inf weight found at "+k)
              
      #name_value = param.eval_metric.get_name_value()
      #for name, value in name_value:
      #    logging.info('Epoch[%d] Validation-%s=%f', param.epoch, name, value)
      loss = param.eval_metric.get_name_value()[1]
      train_acc = param.eval_metric.get_name_value()[0]
      #print(loss)
      #if (np.isnan(loss[1])):
      #    print("Nan loss found")
      #    f = open("nan_loss_weights.txt", "w")
      #    f.write("batch #"+str(global_step[0])+"\n"+str(loss)+"\n"+str(weights_db.keys())+"\n"+str(weights_db))
      #    f.close()
      #    print("Written file at: nan_loss_weights.txt")
      #    exit()
          
      
      
      
      global_step[0]+=1
      mbatch = global_step[0]
      for step in lr_steps:
        if mbatch==step:
          opt.lr *= 0.1
          logging.info('lr change to %f', opt.lr)
          break

      _cb(param)
      if mbatch%1000==0:
        logging.info('lr-batch-epoch: %f %d %d',opt.lr,param.nbatch,param.epoch)

      if mbatch>=0 and mbatch%args.verbose==0:
        acc_list = ver_test(mbatch)
        save_step[0]+=1
        msave = save_step[0]
        do_save = False
        is_highest = False
        if len(acc_list)>0:
          #lfw_score = acc_list[0]
          #if lfw_score>highest_acc[0]:
          #  highest_acc[0] = lfw_score
          #  if lfw_score>=0.998:
          #    do_save = True
          score = sum(acc_list)
          if acc_list[-1]>=highest_acc[-1]:
            if acc_list[-1]>highest_acc[-1]:
              is_highest = True
            else:
              if score>=highest_acc[0]:
                is_highest = True
                highest_acc[0] = score
            highest_acc[-1] = acc_list[-1]
            #if lfw_score>=0.99:
            #  do_save = True
        if is_highest:
          do_save = True
        if args.ckpt==0:
          do_save = False
        elif args.ckpt==2:
          do_save = True
        elif args.ckpt==3:
          msave = 1

        if do_save:
          logging.info('saving %d', msave)
          arg, aux = model.get_params()
          if config.ckpt_embedding:
            all_layers = model.symbol.get_internals()
            _sym = all_layers['fc1_output']
            _arg = {}
            for k in arg:
              if not k.startswith('fc7'):
                _arg[k] = arg[k]
            mx.model.save_checkpoint(prefix, param.epoch + 1, _sym, _arg, aux)
          else:
            mx.model.save_checkpoint(prefix, param.epoch + 1, model.symbol, arg, aux)
        logging.info('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if config.max_steps>0 and mbatch>config.max_steps:
        sys.exit(0)

    epoch_cb = None
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = 999999,
        eval_data          = val_dataiter,
        eval_metric        = eval_metrics,
        kvstore            = args.kvstore,
        optimizer          = opt,
        #optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )

def main():
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

