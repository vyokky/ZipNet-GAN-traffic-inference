import tensorflow as tf


def gaussian_estimator(output, target):
    """Return the TensorFlow expression of gaussian_estimator.

    Parameters
    ----------
    output : tensorflow variable
        A distribution with shape: [batch_size, 2].
    target : tensorflow variable
        A distribution with shape: [batch_size, 1].
    """
    out_mean = tf.reshape(output[:, 0], (-1, 1))
    out_var = tf.reshape(output[:, 1], (-1, 1))

    with tf.name_scope("Gaussian_estimator_loss"):
        gaussian = tf.reduce_sum(tf.squared_difference(out_mean, target) /
                                 (2*tf.square(tf.exp(out_var)))+tf.log(tf.exp(out_var)), reduction_indices=1)
        return tf.reduce_mean(gaussian)


def gaussian_lop(mean, target, mixed_model, alpha, std=None):
    """Return the TensorFlow expression of gaussian_estimator.

    Parameters
    ----------
    mean : tensorflow variable
        A distribution with shape: [batch_size, 1].

    mixed_model: tensorflow variable
        A distribution with shape: [batch_size, 1].

    alpha: tensorflow variable
        mixture weight with shape: [batch_size, 1].

    target : tensorflow variable
        A distribution with shape: [batch_size, 1].

    std: tensorflow variable
        The standard error with shape: [batch_size, 1].
    """
    if std is not None:
        with tf.name_scope("Gaussian_estimator_loss"):
            gaussian = tf.reduce_sum(tf.squared_difference(alpha*mean+(1-alpha)*mixed_model, target) /
                                     (2*tf.square(tf.exp(std)))+tf.log(tf.exp(std)), reduction_indices=1)

            return tf.reduce_mean(gaussian)

    else:
        with tf.name_scope("Gaussian_estimator_loss"):
            gaussian = tf.reduce_sum(tf.squared_difference(alpha*mean+(1-alpha)*mixed_model, target)
                                     , reduction_indices=1)

            return tf.reduce_mean(gaussian)


def gan_dloss(D, G):
    """

    :param D:
    :param G:
    :return:
    """
    with tf.name_scope("gan_dloss"):
        loss = - tf.reduce_sum(tf.log(D) + tf.log(1-G), reduction_indices=1)

        return tf.reduce_mean(loss)


def gan_gloss(G):
    """

    :param D:
    :param G:
    :return:
    """
    with tf.name_scope("gan_gloss"):
        loss = -tf.reduce_sum(tf.log(G), reduction_indices=1)

        return tf.reduce_mean(loss)


def gan_gloss_r(G):
    """

    :param D:
    :param G:
    :return:
    """
    with tf.name_scope("gan_gloss"):
        loss = tf.reduce_sum(tf.log(1-G), reduction_indices=1)

        return tf.reduce_mean(loss)


def wgan_dloss(D, G):
    """

    :param D:
    :param G:
    :return:
    """
    with tf.name_scope("wgan_dloss"):
        loss = - tf.reduce_sum(D - G, reduction_indices=1)

        return tf.reduce_mean(loss)


def wgan_gloss(G):
    """

    :param D:
    :param G:
    :return:
    """
    with tf.name_scope("wgan_gloss"):
        loss = -tf.reduce_sum(G, reduction_indices=1)

        return tf.reduce_mean(loss)


  




