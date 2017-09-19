import tensorlayer as tl
import tensorflow as tf
import numpy as np
import time
import copy
import DataProvider
import sys


def dict_to_one(dp_dict={}):

    """ Input a dictionary, return a dictionary that all items are
    set to one, use for disable dropout, drop-connect layer and so on.

    Parameters
    ----------
    dp_dict : dictionary keeping probabilities date
    """
    return {x: 1 for x in dp_dict}


def sigmoid(x):

    return 1/(1+np.exp(-x))


def modelsaver(network, path, epoch_identifier=None):

    if epoch_identifier:
        ifile = path + '_' + str(epoch_identifier)+'.npz'
    else:
        ifile = path + '.npz'
    tl.files.save_npz(network.all_params, name=ifile)


def customfit(sess, network, cost, train_op, tra_provider, x, y_, acc=None, n_epoch=50,
              print_freq=1, val_provider=None, save_model=-1, tra_kwag=None, val_kwag=None,
              save_path=None, epoch_identifier=None, baseline=10000000000000):
    """
        Train a given network by the given cost function, dataset, n_epoch etc.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        train_op : a TensorFlow optimizer
            like tf.train.AdamOptimizer
        x : placeholder
            for inputs
        y_ : placeholder
            for targets
        cost:  the TensorFlow expression of cost
        acc : the TensorFlow expression of accuracy (or other metric) or None
            if None, would not display the metric
        tra_provider :
            A object of DataProvider for training
        tra_kwag :
            Parameters dic. fed to the tra_provider
        val_provider :
            A object of DataProvider for validation
        val_kwag :
            Parameters dic. fed to the val_provider
        save_model :
            save model mode. 0 -- no save, -1 -- last epoch save, other positive int -- save frequency
        save_path :
            model save path
        epoch_identifier :
            save path + epoch? or not
        n_epoch : int
            the number of training epochs
        print_freq : int
            display the training information every ``print_freq`` epochs
        baseline: early stop first based line
    """

    # assert X_train.shape[0] >= batch_size, "Number of training examples should be bigger than the batch size"
    print("Start training the network ...")

    start_time_begin = time.time()
    for epoch in range(n_epoch):
        start_time = time.time()
        loss_ep = 0;
        n_step = 0

        for batch in tra_provider.feed(**tra_kwag):
            X_train_a, y_train_a = batch
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(network.all_drop)  # enable noise layers
            loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
            loss_ep += loss
            n_step += 1
        loss_ep = loss_ep / n_step

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            if val_provider is not None:
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                train_loss, train_acc, n_batch = 0, 0, 0
                for batch in tra_provider.feed(**tra_kwag):
                    X_train_a, y_train_a = batch
                    dp_dict = dict_to_one(network.all_drop)  # disable noise layers
                    feed_dict = {x: X_train_a, y_: y_train_a}
                    feed_dict.update(dp_dict)
                    if acc is not None:
                        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                        train_acc += ac
                    else:
                        err = sess.run(cost, feed_dict=feed_dict)
                    train_loss += err;
                    n_batch += 1
                print("   train loss: %f" % (train_loss / n_batch))
                # print (train_loss, n_batch)
                if acc is not None:
                    print("   train acc: %f" % (train_acc / n_batch))
                val_loss, val_acc, n_batch = 0, 0, 0

                for batch in val_provider.feed(**val_kwag):
                    X_val_a, y_val_a = batch
                    dp_dict = dict_to_one(network.all_drop)  # disable noise layers
                    feed_dict = {x: X_val_a, y_: y_val_a}
                    feed_dict.update(dp_dict)
                    if acc is not None:
                        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                        val_acc += ac
                    else:
                        err = sess.run(cost, feed_dict=feed_dict)
                    val_loss += err;
                    n_batch += 1
                print("   val loss: %f" % (val_loss / n_batch))
                mean_val_loss = val_loss / n_batch
                if acc is not None:
                    print("   val acc: %f" % (val_acc / n_batch))
            else:
                print("Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep))

        # print(save_model > 0, epoch % save_model == 0, epoch/save_model > 0)

        if save_model > 0 and epoch % save_model == 0:
            if epoch_identifier:
                modelsaver(network=network, path=save_path, epoch_identifier=epoch)
            else:
                modelsaver(network=network, path=save_path, epoch_identifier=None)

        elif save_model == -100:
            if mean_val_loss < baseline:
                modelsaver(network=network, path=save_path, epoch_identifier=None)
                baseline = mean_val_loss

    if save_model == -1:
        modelsaver(network=network, path=save_path, epoch_identifier=None)

    print("Total training time: %fs" % (time.time() - start_time_begin))


def custompredict(sess, network, output_provider, x, fragment_size=1000, output_length=1, y_op=None, out_kwag=None):
    """
        Return the predict results of given non time-series network.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        x : placeholder
            the input
        y_op : placeholder
            the output
        output_provider : DataProvider
        out_kwag :
            Parameter dic. fed to the output_provider
        fragment_size :
            data number predicted for every step
        output_length :
            output size

    """
    dp_dict = dict_to_one(network.all_drop)  # disable noise layers

    if y_op is None:
        y_op = network.outputs
    output_container = []
    gt = []
    banum = 0
    for batch in output_provider.feed(**out_kwag):
        # print banum
        banum += 1
        X_out_a, gt_batch = batch
        # print 'hi', X_out_a.mean()
        fra_num = X_out_a.shape[0] / fragment_size
        offset = X_out_a.shape[0] % fragment_size
        final_output = np.zeros((X_out_a.shape[0], output_length))
        for fragment in xrange(fra_num):
            x_fra = X_out_a[fragment * fragment_size:(fragment + 1) * fragment_size]
            feed_dict = {x: x_fra, }
            feed_dict.update(dp_dict)
            final_output[fragment * fragment_size:(fragment + 1) * fragment_size] = \
                sess.run(y_op, feed_dict=feed_dict).reshape(-1, output_length)

        if offset > 0:
            feed_dict = {x: X_out_a[-offset:], }
            feed_dict.update(dp_dict)
            final_output[-offset:] = sess.run(y_op, feed_dict=feed_dict).reshape(-1, output_length)
        output_container.append(final_output)
        gt.append(gt_batch)
        # print 'hello', final_output.mean()
    return np.vstack(output_container), np.vstack(gt)

