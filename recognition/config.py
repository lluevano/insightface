import numpy as np
import os
from easydict import EasyDict as edict

config = edict()

config.bn_mom = 0.9
config.workspace = 256
config.emb_size = 512
config.ckpt_embedding = True
config.net_se = 0
config.net_act = 'prelu'
config.net_unit = 3
config.net_input = 1
config.net_blocks = [1,4,6,2]
config.net_output = 'E'
config.net_multiplier = 1.0
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
config.ce_loss = True
config.fc7_lr_mult = 1.0
config.fc7_wd_mult = 1.0
config.fc7_no_bias = False
config.max_steps = 0
config.data_rand_mirror = True
config.data_cutoff = False
config.data_color = 0
config.data_images_filter = 0
config.count_flops = True
config.memonger = False #not work now


# network settings
network = edict()

network.r100 = edict()
network.r100.net_name = 'fresnet'
network.r100.num_layers = 100

network.r100fc = edict()
network.r100fc.net_name = 'fresnet'
network.r100fc.num_layers = 100
network.r100fc.net_output = 'FC'

network.r50 = edict()
network.r50.net_name = 'fresnet'
network.r50.num_layers = 50

network.r50v1 = edict()
network.r50v1.net_name = 'fresnet'
network.r50v1.num_layers = 50
network.r50v1.net_unit = 1

network.d169 = edict()
network.d169.net_name = 'fdensenet'
network.d169.num_layers = 169
network.d169.per_batch_size = 64
network.d169.densenet_dropout = 0.0

network.d201 = edict()
network.d201.net_name = 'fdensenet'
network.d201.num_layers = 201
network.d201.per_batch_size = 64
network.d201.densenet_dropout = 0.0

network.y1 = edict()
network.y1.net_name = 'fmobilefacenet'
network.y1.emb_size = 128
network.y1.net_output = 'GDC'

network.y2 = edict()
network.y2.net_name = 'fmobilefacenet'
network.y2.emb_size = 256
network.y2.net_output = 'GDC'
network.y2.net_blocks = [2,8,16,4]

network.m1 = edict()
network.m1.net_name = 'fmobilenet'
network.m1.emb_size = 256
network.m1.net_output = 'GDC'
network.m1.net_multiplier = 1.0

network.m05 = edict()
network.m05.net_name = 'fmobilenet'
network.m05.emb_size = 256
network.m05.net_output = 'GDC'
network.m05.net_multiplier = 0.5

network.mnas = edict()
network.mnas.net_name = 'fmnasnet'
network.mnas.emb_size = 256
network.mnas.net_output = 'GDC'
network.mnas.net_multiplier = 1.0

network.mnas75 = edict()
network.mnas75.net_name = 'fmnasnet'
network.mnas75.emb_size = 256
network.mnas75.net_output = 'GDC'
network.mnas75.net_multiplier = 0.75

network.mnas05 = edict()
network.mnas05.net_name = 'fmnasnet'
network.mnas05.emb_size = 256
network.mnas05.net_output = 'GDC'
network.mnas05.net_multiplier = 0.5

network.mnas025 = edict()
network.mnas025.net_name = 'fmnasnet'
network.mnas025.emb_size = 256
network.mnas025.net_output = 'GDC'
network.mnas025.net_multiplier = 0.25

#custom network definition
network.squeezev1 = edict()
network.squeezev1.net_name = 'fsqueezefacenet_v1'
network.squeezev1.emb_size = 128
network.squeezev1.net_output = 'GDC'   

network.squeezev1_0 = edict()
network.squeezev1_0.net_name = 'fsqueezenet1_0'
network.squeezev1_0.emb_size = 128
network.squeezev1_0.net_output = 'GDC'   
network.squeezev1_0.act_type = 'relu'
network.squeezev1_0.net_act = 'relu'

network.squeezev1_1 = edict()
network.squeezev1_1.net_name = 'fsqueezenet1_1'
network.squeezev1_1.emb_size = 1000
network.squeezev1_1.net_output = 'GDC'   
network.squeezev1_1.act_type = 'relu'
network.squeezev1_1.net_act = 'relu'

network.squeezev1_1_no_pool = edict()
network.squeezev1_1_no_pool.net_name = 'fsqueezenet1_1_no_pool'
network.squeezev1_1_no_pool.emb_size = 1000
network.squeezev1_1_no_pool.net_output = 'GDC'   
network.squeezev1_1_no_pool.act_type = 'relu'
network.squeezev1_1_no_pool.net_act = 'relu'

network.squeezev1_2 = edict()
network.squeezev1_2.net_name = 'fsqueezenet1_2'
network.squeezev1_2.emb_size = 1000
network.squeezev1_2.net_output = 'GDC'   
network.squeezev1_2.act_type = 'relu'
network.squeezev1_2.net_act = 'relu'

network.mobilenetv2 = edict()
network.mobilenetv2.net_name = 'fmobilenetv2'
network.mobilenetv2.emb_size = 1280
network.mobilenetv2.net_output = 'GDC'   
network.mobilenetv2.act_type = 'relu'
network.mobilenetv2.net_act = 'relu'
network.mobilenetv2.width_multiplier = 1.0


network.squeezev2 = edict()
network.squeezev2.net_name = 'fsqueezefacenet_v2'
network.squeezev2.emb_size = 128
network.squeezev2.net_output = 'GDC'


network.shufflev2 = edict()
network.shufflev2.net_name = 'fshufflefacenetv2'
network.shufflev2.emb_size = 128
network.shufflev2.net_output = 'GDC'
network.shufflev2.depth_multiplier = 1.5   #  [0.5, 1.0, 1.5, 2.0]

network.efficient = edict()
network.efficient.net_name = 'fefficientnet'
network.efficient.emb_size = 256
network.efficient.net_output = 'GDC'

network.shufflev1_orig = edict()
network.shufflev1_orig.net_name = 'fshufflenetv1'
network.shufflev1_orig.depth_multiplier = 2.0   #  [0.5, 1.0, 1.5, 2.0] 
network.shufflev1_orig.net_act = 'relu'   
network.shufflev1_orig.fc_type = 'GAP' 
network.shufflev1_orig.emb_size = 2048 #2048 for 2.0 ; 1024 for 1.5

network.mobilefacev1 = edict()
network.mobilefacev1.net_name = 'fmobilefacenetv1'
network.mobilefacev1.emb_size = 128
network.mobilefacev1.net_output =  'GDC'
network.mobilefacev1.net_multiplier = 1.0
network.mobilefacev1.net_act = 'prelu'

network.mobilenetv2_custom = edict()
network.mobilenetv2_custom.net_name='fmobilenetv2_mxnet'
network.mobilenetv2_custom.act_type = 'relu6'
network.mobilenetv2_custom.width_multiplier = 1.0
network.mobilenetv2_custom.emb_size = 1280 

network.vargfacenet = edict()
network.vargfacenet.net_name = 'vargfacenet'
network.vargfacenet.net_multiplier = 1.25
network.vargfacenet.emb_size = 512
network.vargfacenet.net_output='J'

# dataset settings
dataset = edict()

dataset.emore = edict()
dataset.emore.dataset = 'emore'
dataset.emore.dataset_path = '../datasets/faces_emore'
dataset.emore.num_classes = 85742
dataset.emore.image_shape = (112,112,3)
dataset.emore.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.retina = edict()
dataset.retina.dataset = 'retina'
dataset.retina.dataset_path = '../datasets/ms1m-retinaface-t1'
dataset.retina.num_classes = 93431
dataset.retina.image_shape = (112,112,3)
dataset.retina.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.lr_webface = edict()
dataset.lr_webface.dataset = 'lr_webface'
dataset.lr_webface.dataset_path = '../datasets/faces_webface_lr'
dataset.lr_webface.num_classes = 10572
dataset.lr_webface.image_shape = (112, 112, 3)
dataset.lr_webface.val_targets = ['lfw', 'lfw_56', 'lfw_28', 'lfw_14', 'lfw_7']  


dataset.lr_native = edict()
dataset.lr_native.dataset = 'native_lr'
dataset.lr_native.dataset_path = '../datasets/native_lr'
dataset.lr_native.num_classes = 6599
dataset.lr_native.image_shape = (112, 112, 3)
dataset.lr_native.val_targets = ['lfw_28_lr2lr','lfw_21_lr2lr', 'lfw_14_lr2lr', 'lfw_7_lr2lr']  

dataset.lr_scface = edict()
dataset.lr_scface.dataset = 'lr_scface'
dataset.lr_scface.dataset_path = '../datasets/scface_50_lr'
dataset.lr_scface.num_classes = 50
dataset.lr_scface.image_shape = (112, 112, 3)
dataset.lr_scface.val_targets = ['lfw_7','lfw_7_hr2lr_interArea', 'lfw_14', 'lfw_14_hr2lr_interArea']  

dataset.lr_interArea = edict()
dataset.lr_interArea.dataset = 'lr_interArea'
dataset.lr_interArea.dataset_path = '../datasets/lr_interArea'
dataset.lr_interArea.num_classes = 10572
dataset.lr_interArea.image_shape = (112, 112, 3)
dataset.lr_interArea.val_targets = ['lfw_28_hr2lr_interArea','lfw_14_hr2lr_interArea','lfw_7_hr2lr_interArea']  

dataset.lr_interArea_lr2lr = edict()
dataset.lr_interArea_lr2lr.dataset = 'lr_interArea_lr2lr'
dataset.lr_interArea_lr2lr.dataset_path = '../datasets/faces_webface_lr_interArea_lr2lr'
dataset.lr_interArea_lr2lr.num_classes = 10572
dataset.lr_interArea_lr2lr.image_shape = (112, 112, 3)
dataset.lr_interArea_lr2lr.val_targets = ['lfw_28_lr2lr_interArea','lfw_21_lr2lr_interArea', 'lfw_14_lr2lr_interArea','lfw_7_lr2lr_interArea']  

dataset.lr_interCubic_lr2lr = edict()
dataset.lr_interCubic_lr2lr.dataset = 'lr_interCubic_lr2lr'
dataset.lr_interCubic_lr2lr.dataset_path = '../datasets/faces_webface_lr_interCubic_lr2lr'
dataset.lr_interCubic_lr2lr.num_classes = 10572
dataset.lr_interCubic_lr2lr.image_shape = (112, 112, 3)
dataset.lr_interCubic_lr2lr.val_targets = ['lfw_28_lr2lr','lfw_21_lr2lr', 'lfw_14_lr2lr','lfw_7_lr2lr']  

dataset.lr_Cubic_Area = edict()
dataset.lr_Cubic_Area.dataset = 'lr_Cubic_Area'
dataset.lr_Cubic_Area.dataset_path = '../datasets/faces_webface_lr_Cubic_Area'
dataset.lr_Cubic_Area.num_classes = 10572
dataset.lr_Cubic_Area.image_shape = (112, 112, 3)
dataset.lr_Cubic_Area.val_targets = ['lfw_28_hr2lr_interArea','lfw_14_hr2lr_interArea','lfw_7_hr2lr_interArea']  

loss = edict()
loss.softmax = edict()
loss.softmax.loss_name = 'softmax'

loss.nsoftmax = edict()
loss.nsoftmax.loss_name = 'margin_softmax'
loss.nsoftmax.loss_s = 64.0
loss.nsoftmax.loss_m1 = 1.0
loss.nsoftmax.loss_m2 = 0.0
loss.nsoftmax.loss_m3 = 0.0

loss.arcface = edict()
loss.arcface.loss_name = 'margin_softmax'
loss.arcface.loss_s = 64.0
loss.arcface.loss_m1 = 1.0
loss.arcface.loss_m2 = 0.5
loss.arcface.loss_m3 = 0.0

loss.cosface = edict()
loss.cosface.loss_name = 'margin_softmax'
loss.cosface.loss_s = 64.0
loss.cosface.loss_m1 = 1.0
loss.cosface.loss_m2 = 0.0
loss.cosface.loss_m3 = 0.35

loss.combined = edict()
loss.combined.loss_name = 'margin_softmax'
loss.combined.loss_s = 64.0
loss.combined.loss_m1 = 1.0
loss.combined.loss_m2 = 0.3
loss.combined.loss_m3 = 0.2

loss.triplet = edict()
loss.triplet.loss_name = 'triplet'
loss.triplet.images_per_identity = 5
loss.triplet.triplet_alpha = 0.3
loss.triplet.triplet_bag_size = 7200
loss.triplet.triplet_max_ap = 0.0
loss.triplet.per_batch_size = 60
loss.triplet.lr = 0.05

loss.atriplet = edict()
loss.atriplet.loss_name = 'atriplet'
loss.atriplet.images_per_identity = 5
loss.atriplet.triplet_alpha = 0.35
loss.atriplet.triplet_bag_size = 7200
loss.atriplet.triplet_max_ap = 0.0
loss.atriplet.per_batch_size = 60
loss.atriplet.lr = 0.05

# default settings
default = edict()

# default network
default.network = 'r100'
default.pretrained = ''
default.pretrained_epoch = 0
# default dataset
default.dataset = 'emore'
default.loss = 'arcface'
default.frequent = 20
default.verbose = 2000
default.kvstore = 'local'

default.end_epoch = 10000
default.lr = 0.1
default.wd = 0.0005
default.mom = 0.9
default.per_batch_size = 64
default.ckpt = 3
default.lr_steps = '100000,160000,220000'
default.models_root = './models'


def generate_config(_network, _dataset, _loss):
    for k, v in loss[_loss].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in network[_network].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in dataset[_dataset].items():
      config[k] = v
      if k in default:
        default[k] = v
    config.loss = _loss
    config.network = _network
    config.dataset = _dataset
    config.num_workers = 1
    #if 'DMLC_NUM_WORKER' in os.environ:
     # config.num_workers = int(os.environ['DMLC_NUM_WORKER'])

