# -*- coding: utf-8 -*-
"""
训练识别固定长度的验证码
"""
import os

from assembleDataGenerator import *
from inference import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'

# 训练
def train():
    X = tf.placeholder(tf.float32, shape=[None, INPUT_HEIGHT, INPUT_WIDTH])
    Y = tf.placeholder(tf.float32, shape=[None, CODE_LEN * CHAR_SET_LEN])
    # Y = tf.placeholder(tf.float32, shape=[None, CHAR_SET_LEN])
    keep_prob = tf.placeholder(tf.float32)  # dropout
    global_step = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    # 是否在训练阶段
    is_train = tf.placeholder(tf.bool)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)     #正则化项
    output = inference(X, regularizer, keep_prob, is_train)

    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output)
    y_ = tf.reshape(Y, shape=[-1, CHAR_SET_LEN])            #转化为2维数据
    y = tf.reshape(output, shape=[-1, CHAR_SET_LEN])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, axis=1), logits=y)
    loss = tf.reduce_mean(cross_entropy)+tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, DATA_SIZE / BATCH_SIZE, LEARNING_RATE_DECAY)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    predict = tf.reshape(output, [-1, CODE_LEN, NUM_CLASSES])
    max_idx_pred = tf.argmax(predict, 2)
    max_idx_label = tf.argmax(tf.reshape(Y, [-1, CODE_LEN, NUM_CLASSES]), 2)
    correction = tf.equal(max_idx_pred, max_idx_label)
    accuracy = tf.reduce_mean((tf.cast(correction, tf.float32)))

    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        itr = get_train_batch()
        for step in range(NUM_EPOCHS):
            batch_x, batch_y = next(itr)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0, is_train: True})
            print('Epoch %s/%s, loss=%.2f' % (step, NUM_EPOCHS, loss_))

            # 每100 step计算一次准确率
            if step % 100 == 0 and step != 0:
                # batch_x_val, batch_y_val = next(itr)
                batch_x_val, batch_y_val, txts = get_val_data()
                acc, loss_ = sess.run([accuracy,loss], feed_dict={X: batch_x_val, Y: batch_y_val, keep_prob: 1., is_train: False})

                # for i in range(batch_y_val.shape[0]):
                #     print("The label is {}, the prediction is {}".format(vec2text(batch_y_val[i]), vec2text(pred[i])))

                print("Epoch %s, on validation, loss = %s, accuracy = %s" % (step, loss_, acc))
                # 如果准确率大80%,保存模型,完成训练
                if acc > 0.8:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
