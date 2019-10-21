import sys
import LayerExtension
import tensorflow as tf
import tensorlayer as tl


def zipper(x, input_x, input_y, act=tf.nn.leaky_relu, reuse = False, name = 'zipper', is_train=True,
           observations=6, downscale=4):

    res3d_map = 6
    res2d_map = 36
    temporal_stride = min(observations, 3)

    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = tl.layers.InputLayer(x, name='input_layer' )
        network = tl.layers.TransposeLayer(network, (0, 2, 3, 1), name='trans')

        ## Downsample
        network = tl.layers.PoolLayer(network, ksize=[1,  downscale, downscale, 1], strides=[1, downscale, downscale,1],
                                      padding='VALID', pool=tf.nn.avg_pool, name='pool_layer')
        network = tl.layers.TransposeLayer(network, (0, 3, 1, 2), name='trans2')
        network = tl.layers.ReshapeLayer(network, shape=(-1, observations, input_x/downscale,
                                                         input_y/downscale, 1), name='reshape')
        ## 3D-upscale module
        network = LayerExtension.DeConv3dLayer(layer= network , shape=[1, 3, 3, res3d_map, 1],
                                               output_shape=[-1, observations, input_x/2, input_y/2, res3d_map],
                 strides=[1, 1, 2, 2, 1], padding='SAME',  name='decnn3d_layer1')
        network = tl.layers.BatchNormLayer(network, name = '3d_bn1', act = act,is_train = is_train)
        network = tl.layers.Conv3dLayer(network, shape=[temporal_stride, 3, 3, res3d_map, res3d_map],
                                          strides=[1, 1, 1, 1, 1], padding='SAME', name='cnn3d_layer1',  act=act)
        network = tl.layers.Conv3dLayer(network, shape=[temporal_stride, 3, 3, res3d_map, res3d_map],
                                          strides=[1, 1, 1, 1, 1], padding='SAME', name='cnn3d_layer2',  act=act)
        network = tl.layers.Conv3dLayer(network, shape=[temporal_stride, 3, 3, res3d_map, res3d_map],
                                          strides=[1, 1, 1, 1, 1], padding='SAME', name='cnn3d_layer3',  act=act)
        network = LayerExtension.DeConv3dLayer(layer= network , shape=[1, 3, 3, res3d_map, res3d_map],
                                               output_shape=[-1, observations, input_x, input_y, res3d_map],
                 strides=[1, 1, 2, 2, 1], padding='SAME',  name='decnn3d_layer2')
        network = tl.layers.BatchNormLayer(network, name = '3d_bn2', act = act,is_train = is_train)
        network = tl.layers.Conv3dLayer(network, shape=[temporal_stride, 3, 3, res3d_map, res3d_map],
                                          strides=[1, 1, 1, 1, 1], padding='SAME', name='cnn3d_layer4',  act=act)
        network = tl.layers.Conv3dLayer(network, shape=[temporal_stride, 3, 3, res3d_map, res3d_map],
                                          strides=[1, 1, 1, 1, 1], padding='SAME', name='cnn3d_layer5',  act=act)
        network = tl.layers.Conv3dLayer(network, shape=[temporal_stride, 3, 3, res3d_map, res3d_map],
                                          strides=[1, 1, 1, 1, 1], padding='SAME', name='cnn3d_layer6',  act=act)
        
        ## adjustments
        network = LayerExtension.TransposeLayer(network, (0, 2, 3, 1, 4), name='trans3')
        network = tl.layers.ReshapeLayer(network, shape=(-1, input_x, input_y, observations*res3d_map), name='reshape2')
        network = tl.layers.Conv2dLayer(network, shape=[4, 4, observations*res3d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='cnn2d_layer_2d_ini1')
        network_up = tl.layers.BatchNormLayer(network, name = 'starter', act = act,is_train = is_train)
        ## zipper start

        z1 = tl.layers.Conv2dLayer(network_up, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z1',  act=tl.activation.identity)
        bn1 = tl.layers.BatchNormLayer(z1, name = 'bn1', act = act,is_train = is_train)
        z2 = tl.layers.Conv2dLayer(bn1, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z2',  act=tl.activation.identity)
        bn2 = tl.layers.BatchNormLayer(z2, name = 'bn2', act = act,is_train = is_train)
        z3 = tl.layers.Conv2dLayer(bn2, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z3',  act=tl.activation.identity)
        bn3 = tl.layers.BatchNormLayer(z3, name = 'bn3', act = act,is_train = is_train)
        connect1 = tl.layers.ElementwiseLayer([bn1, bn3], combine_fn=tf.add, name='c1')
        z4 = tl.layers.Conv2dLayer(connect1, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z4',  act=tl.activation.identity)
        bn4 = tl.layers.BatchNormLayer(z4, name = 'bn4', act = act,is_train = is_train)
        connect2 = tl.layers.ElementwiseLayer([bn2, bn4], combine_fn=tf.add, name='c2')
        z5 = tl.layers.Conv2dLayer(connect2, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z5',  act=tl.activation.identity)
        bn5 = tl.layers.BatchNormLayer(z5, name = 'bn5', act = act,is_train = is_train)
        connect3 = tl.layers.ElementwiseLayer([bn3, bn5], combine_fn=tf.add, name='c3')
        z6 = tl.layers.Conv2dLayer(connect3, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z6',  act=tl.activation.identity)
        bn6 = tl.layers.BatchNormLayer(z6, name = 'bn6', act = act,is_train = is_train)
        connect4 = tl.layers.ElementwiseLayer([bn4, bn6], combine_fn=tf.add, name='c4')   
        z7 = tl.layers.Conv2dLayer(connect4, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z7',  act=tl.activation.identity)
        bn7 = tl.layers.BatchNormLayer(z7, name = 'bn7', act = act,is_train = is_train)
        connect5 = tl.layers.ElementwiseLayer([bn5, bn7], combine_fn=tf.add, name='c5')
        z8 = tl.layers.Conv2dLayer(connect5, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z8',  act=tl.activation.identity)
        bn8 = tl.layers.BatchNormLayer(z8, name = 'bn8', act = act,is_train = is_train)
        connect6 = tl.layers.ElementwiseLayer([bn6, bn8], combine_fn=tf.add, name='c6')
        z9 = tl.layers.Conv2dLayer(connect6, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z9',  act=tl.activation.identity)
        bn9 = tl.layers.BatchNormLayer(z9, name = 'bn9', act = act,is_train = is_train)
        connect7 = tl.layers.ElementwiseLayer([bn7, bn9], combine_fn=tf.add, name='c7')
        z10 = tl.layers.Conv2dLayer(connect7, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z10',  act=tl.activation.identity)
        bn10 = tl.layers.BatchNormLayer(z10, name = 'bn10', act = act,is_train = is_train)
        connect8 = tl.layers.ElementwiseLayer([bn8, bn10], combine_fn=tf.add, name='c8')     
        z11 = tl.layers.Conv2dLayer(connect8, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z11',  act=tl.activation.identity)
        bn11 = tl.layers.BatchNormLayer(z11, name = 'bn11', act = act,is_train = is_train)
        connect9 = tl.layers.ElementwiseLayer([bn9, bn11], combine_fn=tf.add, name='c9') 
        z12 = tl.layers.Conv2dLayer(connect9, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z12',  act=tl.activation.identity)
        bn12 = tl.layers.BatchNormLayer(z12, name = 'bn12', act = act,is_train = is_train)
        connect10 = tl.layers.ElementwiseLayer([bn10, bn12], combine_fn=tf.add, name='c10')  
        z13 = tl.layers.Conv2dLayer(connect10, shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='z13',  act=tl.activation.identity)
        bn13 = tl.layers.BatchNormLayer(z13, name = 'bn13', act = act, is_train = is_train)
        connect11 = tl.layers.ElementwiseLayer([bn11, bn13], combine_fn=tf.add, name='c11')
        network = tl.layers.Conv2dLayer(connect11 , shape=[4, 4, res2d_map, res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='cnn2d_layer1',  act=tl.activation.identity)
        bn13 = tl.layers.BatchNormLayer(network, name = 'bn_whole', act = act, is_train = is_train)
        ## final merge
        network = tl.layers.ElementwiseLayer([bn13, network_up], combine_fn=tf.add, name='global')
        network = tl.layers.Conv2dLayer(network , shape=[4, 4, res2d_map, 2*res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='cnn2d_layer2',  act=act)
        network = tl.layers.Conv2dLayer(network, shape=[4, 4, 2*res2d_map, 3*res2d_map],
                                          strides=[1, 1, 1, 1], padding='SAME', name='cnn2d_layer3',  act=act)
        network = tl.layers.Conv2dLayer(network, shape=[4, 4, 3*res2d_map, 1],
                                          strides=[1, 1, 1, 1], padding='SAME', name='cnn2d_layer4',  act=tl.activation.identity)
        network = tl.layers.FlattenLayer(network) 
        return network
