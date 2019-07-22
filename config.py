from easydict import EasyDict as edict

__C  = edict()

cfg = __C

#Train Options
__C.TRAIN = edict()

__C.TRAIN.ANNO_PATH = ''
__C.TRAIN.INPUTSIZE = 224
__C.TRAIN.LEARN_RATE_INIT = 1e-4
__C.TRAIN.BATCHSIZE = 16
__C.TRAIN.INITIAL_WEIGHT = 'ckpt'
__C.TRAIN.DATAAUG = False
__C.TRAIN.EPOCH = 60
__C.TRAIN.DROPOUT = 0.8
__C.TRAIN.NUMCLASS = 8


#Test Options
__C.TEST = edict()

__C.TEST.ANNO_PATH = ''
__C.TEST.INPUTSIZE = 224
__C.TEST.BATCHSIZE = 16
__C.TEST.WEIGHT_FILE = None
__C.TEST.DATAAUG = False
