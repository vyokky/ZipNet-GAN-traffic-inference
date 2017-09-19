import tensorflow as tf
import tensorlayer as tl


def discriminator(inputs, input_x, input_y, name='D', ini_feature=36, reuse=False,
                  act=lambda x: tl.act.lrelu(x, 0.1), is_train=True):

    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = tl.layers.InputLayer(inputs, name='input_layer' + name)
        network = tl.layers.ReshapeLayer(network, shape=(-1, input_x, input_y, 1), name='reshape' + name)
        network = tl.layers.Conv2dLayer(network, shape=[3, 3, 1, ini_feature],
                                        strides=[1, 1, 1, 1], padding='VALID', name='cnn2d_layer1' + name,
                                        act=tl.activation.identity)
        network = tl.layers.BatchNormLayer(network, name=name + 'bn1', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(network, shape=[3, 3, ini_feature, ini_feature * 2],
                                        strides=[1, 2, 2, 1], padding='VALID', name='cnn2d_layer2' + name,
                                        act=tl.activation.identity)
        network = tl.layers.BatchNormLayer(network, name=name + 'bn2', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(network, shape=[3, 3, ini_feature * 2, ini_feature * 2],
                                        strides=[1, 1, 1, 1], padding='VALID', name='cnn2d_layer3' + name,
                                        act=tl.activation.identity)
        network = tl.layers.BatchNormLayer(network, name=name + 'bn3', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(network, shape=[3, 3, ini_feature * 2, ini_feature * 4],
                                        strides=[1, 2, 2, 1], padding='VALID', name='cnn2d_layer4' + name,
                                        act=tl.activation.identity)
        network = tl.layers.BatchNormLayer(network, name=name + 'bn4', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(network, shape=[3, 3, ini_feature * 4, ini_feature * 4],
                                        strides=[1, 1, 1, 1], padding='VALID', name='cnn2d_layer5' + name,
                                        act=tl.activation.identity)
        network = tl.layers.BatchNormLayer(network, name=name + 'bn5', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(network, shape=[3, 3, ini_feature * 4, ini_feature * 8],
                                        strides=[1, 2, 2, 1], padding='VALID', name='cnn2d_layer6' + name,
                                        act=tl.activation.identity)
        #         network = tl.layers.BatchNormLayer(network, name = name + 'bn6', act = act, is_train = is_train)
        network = tl.layers.FlattenLayer(network, name='flatten' + name)
        network = tl.layers.DenseLayer(network, n_units=1,
                                       act=tf.nn.sigmoid,
                                       name='output_layer' + name)

        return network