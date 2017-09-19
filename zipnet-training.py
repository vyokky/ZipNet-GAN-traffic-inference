import toolbox.NetFlow as nf
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from toolbox import DataProvider
import argparse
import toolbox.CostFunc as lost

sess = tf.InteractiveSession()


def get_arguments():

    parser = argparse.ArgumentParser(description='Train a neural network\
                                     for fine-grained mobile traffic inference from coarse-grained counterparts - \
                                     network input = coarse-grained traffic measurements \
                                     network target = fined-grained traffic measurements')
    parser.add_argument('--datadir',
                        type=str,
                        default='./data/',
                        help='this is the directory of the training samples')
    parser.add_argument('--batchsize',
                        type=int,
                        default=100,
                        help='The batch size of training examples')
    parser.add_argument('--epoch',
                        type=int,
                        default=50,
                        help='The number of epoches.')
    parser.add_argument('--save_model',
                        type=str,
                        default='ultra_zipper.npz',
                        help='Save the learnt model: \
                            0 -- not to save the learnt model parameters;\
                            n (n>0) -- to save the model params every n steps;\
                            -1 -- only save the learnt model params at the end of training.\
                             n = 1 by default')
    parser.add_argument('--model_path',
                        type=str,
                        default='./',
                        help='Saved model path')
    parser.add_argument('--model_name',
                        type=str,
                        default='zipnet',
                        help='Saved model name')
    parser.add_argument('--pre_model_path',
                        type=str,
                        default=None,
                        help='Pretrained model path')
    parser.add_argument('--pre_model_name',
                        type=str,
                        default=None,
                        help='Pretrained model name')
    parser.add_argument('--mean',
                        type=float,
                        default=0,
                        help='mean value for data normalisation: \
                            (data-mean)/std')
    parser.add_argument('--std',
                        type=float,
                        default=1,
                        help='standard deviation value for data normalisation: \
                            (data-mean)/std')
    parser.add_argument('--observations',
                        type=int,
                        default=6,
                        help='temporal length of input')
    parser.add_argument('--lr',
                        type=int,
                        default=0.0001,
                        help='learning rate of the model')
    parser.add_argument('--downscale',
                        type=int or str,
                        default=4,
                        help="downscale of coarse-grained measurement, \
                              downscale = [2, 4, 10, 'mix'] are supported")
    parser.add_argument('--framebatch',
                        type=int,
                        default=1,
                        help="number of frame scanned in a training batch")
    parser.add_argument('--input_x',
                        type=int,
                        default=80,
                        help="spatial length of input of x axis")
    parser.add_argument('--input_y',
                        type=int,
                        default=80,
                        help="spatial length of input of y axis")
    return parser.parse_args()

args = get_arguments()


def load_dataset():

    tra_set = (np.load(args.datadir + 'milan_tra.npy') - args.mean) / args.std
    val_set = (np.load(args.datadir + 'milan_val.npy') - args.mean) / args.std

    print('training set:', tra_set.shape)
    print('validation set:', val_set.shape)

    return tra_set, val_set

tra_set, val_set = load_dataset()

observations = args.observations
input_size = (args.input_x, args.input_y, observations)
framebatch = args.framebatch
output_size = (args.input_x, args.input_y)
downscale = args.downscale

tra_kwag = {
    'inputs': tra_set,
    'framebatch': framebatch}

val_kwag = {
    'inputs': val_set,
    'framebatch': framebatch}

if args.downscale == 'mix':
    tra_provider = DataProvider.SpecialSuperResolutionProvider(stride=(1, 1), input_size=input_size,
                           output_size=output_size,  batchsize=args.batchsize,  shuffle=True)
    val_provider = DataProvider.SpecialSuperResolutionProvider(stride=(1, 1), input_size=input_size,
                           output_size=output_size,  batchsize=-1,  shuffle=False)
else:
    tra_provider = DataProvider.SuperResolutionProvider(stride=(1, 1), input_size=input_size,
                                                        output_size=output_size, batchsize=args.batchsize, shuffle=True)
    val_provider = DataProvider.SuperResolutionProvider(stride=(1, 1), input_size=input_size,
                                                        output_size=output_size, batchsize=-1, shuffle=False)

x = tf.placeholder(tf.float32, shape=[None, observations, args.input_x, args.input_y], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, args.input_x * args.input_y], name='y_')

if args.downscale == 2:
    from architecture.up2 import zipper
if args.downscale == 4:
    from architecture.up4 import zipper
if args.downscale == 10:
    from architecture.up10 import zipper
if args.downscale == 'mix':
    from architecture.mix import zipper


network = zipper(x, downscale, args.input_x, args.input_y,  is_train=True, observation=args.observations)
y = network.outputs
cost = tl.cost.mean_squared_error(y, y_)
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.9, beta2=0.999,
                                  epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# initialize all variables
sess.run(tf.initialize_all_variables())

if args.pre_model_path is not None:
    load_params = tl.files.load_npz(path=args.pre_model_path, name=args.pre_model_name)
    tl.files.assign_params(sess, load_params, network)

print 'set done'

nf.customfit(sess=sess, network=network, cost=cost, train_op=train_op, tra_provider=tra_provider, x=x, y_=y_, acc=None,
             n_epoch=args.epoch, print_freq=1, val_provider=val_provider, save_model=1, tra_kwag=tra_kwag, val_kwag=val_kwag,
             save_path=args.model_path + args.model_name, epoch_identifier=None)
