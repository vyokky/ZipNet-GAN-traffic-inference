import toolbox.NetFlow as nf
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from toolbox import DataProvider
import argparse
from architecture import discriminator

sess = tf.InteractiveSession()


def get_arguments():

    parser = argparse.ArgumentParser(description='MTSR inference')
    parser.add_argument('--datadir',
                        type=str,
                        default='./data/',
                        help='this is the directory of the training samples')
    parser.add_argument('--prediction_file',
                        type=str,
                        default='./predictions.npy',
                        help='Saved prediction path and file name')
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
    parser.add_argument('--downscale',
                        type=int or str,
                        default=4,
                        help="downscale of coarse-grained measurement, \
                              downscale = [2, 4, 10, 'mix'] are supported")
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


test_set = (np.load(args.datadir + 'milan_tra.npy') - args.mean) / args.std
print('test set:', test_set.shape)

observations = args.observations
input_size = (args.input_x, args.input_y, observations)
output_size = (args.input_x, args.input_y)
downscale = args.downscale

test_provider = DataProvider.MoverProvider(length=observations)


test_val = {
    'inputs':test_set_y,
    'targets':test_set_y,
    'special': True if downscale == 'mix' else False,
    'keepdims':True}

if args.downscale == 2:
    from up2 import zipper
if args.downscale == 4 or 'mix':
    from up4 import zipper
if args.downscale == 10:
    from up10 import zipper

x = tf.placeholder(tf.float32, shape=[None, observations, args.input_x, args.input_y], name='x')
network = zipper(x, downscale, args.input_x, args.input_y,  is_train=True, observation=args.observations)


if args.pre_model_path is not None:
    load_params = tl.files.load_npz(path=args.pre_model_path, name=args.pre_model_name)
    tl.files.assign_params(sess, load_params, network)

print 'set done'

prediction = nf.custompredict(sess=sess, network=network, output_provider=test_provider, x=x, fragment_size=1,
                              output_length=6400, y_op=None, out_kwag=test_val)

prediction = prediction[0]*args.std+args.mean
np.save(args.prediction_file, prediction)
