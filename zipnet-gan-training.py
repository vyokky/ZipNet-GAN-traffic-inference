import toolbox.NetFlow as nf
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from toolbox import DataProvider
import argparse
from architecture import discriminator
import toolbox.CostFunc as lost

sess = tf.InteractiveSession()


def get_arguments():

    parser = argparse.ArgumentParser(description='Train a GAN\
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
                        default=50000,
                        help='The number of epoches.')
    parser.add_argument('--d_pretrain',
                        type=int,
                        default=100,
                        help='The number of epoches for pretraining discriminator.')
    parser.add_argument('--d_step',
                        type=int,
                        default=1,
                        help='The number of step for discriminator training in a epoch.')
    parser.add_argument('--g_step',
                        type=int,
                        default=1,
                        help='The number of step for geenrator training in a epoch.')
    parser.add_argument('--save_d',
                        type=int,
                        default=-1,
                        help='Save the learnt discriminator: \
                            0 -- not to save the learnt model parameters;\
                            n (n>0) -- to save the model params every n steps;\
                            -1 -- only save the learnt model params at the end of training.\
                             n = 1 by default')
    parser.add_argument('--save_g',
                        type=int,
                        default=-1,
                        help='Save the learnt generator: \
                            0 -- not to save the learnt model parameters;\
                            n (n>0) -- to save the model params every n steps;\
                            -1 -- only save the learnt model params at the end of training.\
                             n = 1 by default')
    parser.add_argument('--save_model_path',
                        type=str,
                        default='./',
                        help='Saved model path')
    parser.add_argument('--save_g_name',
                        type=str,
                        default='generator',
                        help='Saved model name')
    parser.add_argument('--save_d_name',
                        type=str,
                        default='discriminator',
                        help='Saved model name')
    parser.add_argument('--pre_model_path',
                        type=str,
                        default=None,
                        help='Pretrained model path')
    parser.add_argument('--pre_g_name',
                        type=str,
                        default=None,
                        help='Pretrained generator name')
    parser.add_argument('--pre_d_name',
                        type=str,
                        default=None,
                        help='Pretrained discriminator name')
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
    parser.add_argument('--d_lr',
                        type=int,
                        default=0.0001,
                        help='learning rate of the discriminator')
    parser.add_argument('--g_lr',
                        type=int,
                        default=0.0001,
                        help='learning rate of the generator')
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
    'inputs':tra_set,
    'framebatch': framebatch,
    'batch_num': 1,
    'keepdims': True}

val_kwag = {
    'inputs':val_set,
    'framebatch': framebatch,
    'batch_num': 1,
    'keepdims': True}

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


g_x = tf.placeholder(tf.float32, shape=[None, observations,  args.input_x, args.input_y], name='gx')
g_y = tf.placeholder(tf.float32, shape=[None, args.input_x * args.input_y], name='gy')

d_x = tf.placeholder(tf.float32, shape=[None, args.input_x * args.input_y], name='dx')
d_y = tf.placeholder(tf.float32, shape=[None, 1], name='dy')

if args.downscale == 2:
    from architecture.up2 import zipper
if args.downscale == 4:
    from architecture.up4 import zipper
if args.downscale == 10:
    from architecture.up10 import zipper
if args.downscale == 'mix':
    from architecture.mix import zipper


G = zipper(g_x)
D = discriminator(d_x, reuse=False)
D_generate = discriminator(G.outputs, reuse=True)

G_params = G.all_params
D_params = D.all_params

# G_lost = l_weight*lost.gan_gloss(D_generate.outputs) + tl.cost.mean_squared_error(G.outputs, g_y)
G_lost = (1 + 2*lost.gan_gloss(D_generate.outputs)) * tl.cost.mean_squared_error(G.outputs, g_y)
D_lost = lost.gan_dloss(D.outputs, D_generate.outputs)
mse = tl.cost.mean_squared_error(G.outputs, g_y)


G_op = tf.train.AdamOptimizer(learning_rate=args.g_lr, beta1=0.9, beta2=0.999,
                             epsilon=1e-08, use_locking=False).minimize(G_lost, var_list=G_params) #0.0001
#
D_op = tf.train.AdamOptimizer(learning_rate=args.d_lr, beta1=0.9, beta2=0.999, #0.001
                             epsilon=1e-08, use_locking=False).minimize(D_lost, var_list=D_params )

# initialize all variables
sess.run(tf.initialize_all_variables())
if args.pre_g_name is not None:
    load_params = tl.files.load_npz(path=args.pre_model_path, name=args.pre_g_name)
    tl.files.assign_params(sess, load_params, G)
if args.pre_d_name is not None:
    load_params = tl.files.load_npz(path=args.pre_model_path, name=args.pre_d_name)
    tl.files.assign_params(sess, load_params, D)

print 'set done'


g_epoch = args.g_step
joint_epoch = args.epoch

for e in range(joint_epoch):
    print 'Joint epoch', e + 1
    if e == 0:
        d_epoch = args.d_pretrain
    else:
        d_epoch = args.d_step

    nf.customfit(sess=sess, network=D_generate, cost=D_lost, train_op=D_op, tra_provider=tra_provider, x=g_x,
                      y_=d_x, acc=None, n_epoch=d_epoch,
                      print_freq=1, val_provider=None, save_model=args.save_d, tra_kwag=tra_kwag, val_kwag=None,
                      save_path=args.save_model_path + args.save_d_name, epoch_identifier=None)

    nf.customfit(sess=sess, network=G, cost=G_lost, train_op=G_op, tra_provider=tra_provider, x=g_x, y_=g_y, acc=mse,
                 n_epoch=g_epoch,
                 print_freq=1, val_provider=val_provider, save_model=args.save_g, tra_kwag=tra_kwag, val_kwag=val_kwag,
                 save_path=args.save_model_path + args.save_g_name, epoch_identifier=None)
