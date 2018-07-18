import tensorflow as tf

from utils.CONFIG import *
from utils.base_op import *

def inference(X, regularizer, keep_prob, is_train):
    x = tf.reshape(X, shape=[-1, INPUT_HEIGHT, INPUT_WIDTH, 1])

    """
    conv1_1 = conv_op(x, "conv1_1", kh=5, kw=5, sh=1, sw=1, n_out=64, is_train=is_train)
    conv1_2 = conv_op(conv1_1, "conv1_2", kh=5, kw=5, sh=1, sw=1, n_out=64, is_train=is_train)
    pool1 = pool_op(conv1_2, var_scope="pool1", kh=2, kw=2, sh=2, sw=2)

    conv2_1 = conv_op(pool1, "conv2_1", kh=3, kw=3, sh=1, sw=1, n_out=128, is_train=is_train)
    conv2_2 = conv_op(conv2_1, "conv2_2", kh=3, kw=3, sh=1, sw=1, n_out=128, is_train=is_train)
    pool2 = pool_op(conv2_2, var_scope="pool2", kh=2, kw=2, sh=2, sw=2)

    conv3_1 = conv_op(pool2, "conv3_1", kh=3, kw=3, sh=1, sw=1, n_out=256, is_train=is_train)
    conv3_2 = conv_op(conv3_1, "conv3_2", kh=3, kw=3, sh=1, sw=1, n_out=256, is_train=is_train)
    conv3_3 = conv_op(conv3_2, "conv3_3", kh=3, kw=3, sh=1, sw=1, n_out=256, is_train=is_train)
    conv3_4 = conv_op(conv3_3, "conv3_4", kh=3, kw=3, sh=1, sw=1, n_out=256, is_train=is_train)
    pool3 = pool_op(conv3_4, var_scope="pool3", kh=2, kw=2, sh=2, sw=2)

    conv4_1 = conv_op(pool3, "conv4_1", kh=3, kw=3, sh=1, sw=1, n_out=512, is_train=is_train)
    conv4_2 = conv_op(conv4_1, "conv4_2", kh=3, kw=3, sh=1, sw=1, n_out=512, is_train=is_train)
    conv4_3 = conv_op(conv4_2, "conv4_3", kh=3, kw=3, sh=1, sw=1, n_out=512, is_train=is_train)
    conv4_4 = conv_op(conv4_3, "conv4_4", kh=3, kw=3, sh=1, sw=1, n_out=512, is_train=is_train)
    pool4 = pool_op(conv4_4, var_scope="pool4", kh=2, kw=2, sh=2, sw=2)

    conv5_1 = conv_op(pool4, "conv5_1", kh=3, kw=3, sh=1, sw=1, n_out=512, is_train=is_train)
    conv5_2 = conv_op(conv5_1, "conv5_2", kh=3, kw=3, sh=1, sw=1, n_out=512, is_train=is_train)
    conv5_3 = conv_op(conv5_2, "conv5_3", kh=3, kw=3, sh=1, sw=1, n_out=512, is_train=is_train)
    conv5_4 = conv_op(conv5_3, "conv5_4", kh=3, kw=3, sh=1, sw=1, n_out=512, is_train=is_train)
    pool5 = pool_op(conv5_4, var_scope="pool5", kh=2, kw=2, sh=2, sw=2)

    flattened = pool5.shape[1].value * pool5.shape[2].value * pool5.shape[3].value
    reshaped = tf.reshape(pool5, [-1, flattened])

    fc6 = fc_op(reshaped, "fc6", 2048, regularizer)
    fc6_drop = tf.nn.dropout(fc6, keep_prob=keep_prob)

    fc7 = fc_op(fc6_drop, "fc7", 1024, regularizer)
    fc7_drop = tf.nn.dropout(fc7, keep_prob=keep_prob)

    fc8 = fc_op(fc7_drop, "fc8", CODE_LEN * CHAR_SET_LEN, regularizer)
    """

    conv1 = conv_op(x, "conv1", kh=5, kw=5, sh=1, sw=1, n_out=32, is_train=is_train, movAve_decay=MOVING_AVERAGE_DECAY, bn_eps=BN_EPS)
    pool1 = pool_op(conv1, var_scope="pool1", kh=2, kw=2, sh=2, sw=2)

    conv2 = conv_op(pool1, "conv2", kh=5, kw=5, sh=1, sw=1, n_out=64, is_train=is_train, movAve_decay=MOVING_AVERAGE_DECAY, bn_eps=BN_EPS)
    pool2 = pool_op(conv2, var_scope="pool2", kh=2, kw=2, sh=2, sw=2)

    conv3 = conv_op(pool2, "conv3", kh=3, kw=3, sh=1, sw=1, n_out=64, is_train=is_train, movAve_decay=MOVING_AVERAGE_DECAY, bn_eps=BN_EPS)
    pool3 = pool_op(conv3, var_scope="pool3", kh=2, kw=2, sh=2, sw=2)

    conv4 = conv_op(pool3, "conv4", kh=3, kw=3, sh=1, sw=1, n_out=64, is_train=is_train, movAve_decay=MOVING_AVERAGE_DECAY, bn_eps=BN_EPS)
    pool4 = pool_op(conv4, var_scope="pool4", kh=2, kw=2, sh=2, sw=2)

    conv5 = conv_op(pool4, "conv5", kh=3, kw=3, sh=1, sw=1, n_out=64, is_train=is_train, movAve_decay=MOVING_AVERAGE_DECAY, bn_eps=BN_EPS)
    pool5 = pool_op(conv5, var_scope="pool5", kh=2, kw=2, sh=2, sw=2)

    conv6 = conv_op(pool5, "conv6", kh=3, kw=3, sh=1, sw=1, n_out=64, is_train=is_train, movAve_decay=MOVING_AVERAGE_DECAY, bn_eps=BN_EPS)
    pool6 = pool_op(conv6, var_scope="pool6", kh=2, kw=2, sh=2, sw=2)

    flattened = pool6.shape[1].value * pool6.shape[2].value * pool6.shape[3].value
    reshaped = tf.reshape(pool6, [-1, flattened])

    fc7 = fc_op(reshaped, "fc7", 1024, regularizer, False)
    fc7_drop = tf.nn.dropout(fc7, keep_prob=keep_prob)

    fc8 = fc_op(fc7_drop, "fc8", CODE_LEN * CHAR_SET_LEN, regularizer, True)
    out = tf.nn.dropout(fc8, keep_prob=keep_prob)

    # fc8 = _fc_op(fc7_drop, "fc8", CHAR_SET_LEN, regularizer, True)
    # out = tf.nn.dropout(fc8, keep_prob=keep_prob)

    return tf.sigmoid(out)