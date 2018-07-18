# -*- coding: utf-8 -*-
"""
识别不定长验证码
"""
from assembleDataGenerator_m import *
from inference_m import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'

def train():
    X = tf.placeholder(tf.float32, [None, INPUT_WIDTH, INPUT_HEIGHT])
    # 定义ctc_loss需要的稀疏矩阵
    Y = tf.sparse_placeholder(tf.int32)
    is_train = tf.placeholder(tf.bool)

    # 1维向量 序列长度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 1000, LEARNING_RATE_DECAY)
    logits = inference(X, seq_len, is_train)

    loss = tf.nn.ctc_loss(labels=Y, inputs=logits, sequence_length=seq_len)
    loss = tf.reduce_mean(loss)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=True)
    # acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), Y))

    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        itr = get_train_batch()
        for epoch in range(NUM_EPOCHS):
            train_inputs, train_targets, train_seq_len = next(itr)
            # preds = sess.run(logits, {X: train_inputs, seq_len: train_seq_len})
            _,loss_,lr_,logits_ = sess.run([optimizer, loss, learning_rate, logits], {X: train_inputs, Y: train_targets, seq_len: train_seq_len, is_train: True})
            print('Epoch %s/%s, loss=%s, lr=%s' % (epoch, NUM_EPOCHS, loss_, lr_))

            # 每100 step计算一次准确率
            if epoch % 100 == 0 and epoch != 0:
                val_inputs, val_targets, val_seq_len = get_val_data()
                logits_ = sess.run([logits], feed_dict={X: val_inputs, seq_len: val_seq_len, is_train: False})
                loss_, decoded_, log_prob_, lr_ = sess.run([loss, decoded, log_prob, learning_rate], feed_dict={X: val_inputs, Y: val_targets, seq_len: val_seq_len, is_train: False})

                report_accuracy(decoded_[0], val_targets)
                print("Epoch %s, on validation, loss = %s, lr = %s" % (epoch, loss_, lr_))
                # 如果准确率大80%,保存模型,完成训练
                # if acc_ > 0.8:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=epoch)


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()