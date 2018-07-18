"""测试模型"""
import tensorflow as tf
from inference import *
import time
from utils.CONFIG import *
from assembleDataGenerator import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'

def evaluate():

    X = tf.placeholder(tf.float32, shape=[None, INPUT_HEIGHT, INPUT_WIDTH])
    Y = tf.placeholder(tf.float32, shape=[None, CODE_LEN * NUM_CLASSES])
    keep_prob = tf.placeholder(tf.float32)  # dropout
    # 是否在训练阶段
    is_train = tf.placeholder(tf.bool)


    y=inference(X, None, keep_prob, is_train)

    predict = tf.reshape(y, [-1, CODE_LEN, CHAR_SET_LEN])
    max_idx_pre = tf.argmax(predict, 2)
    max_idx_label = tf.argmax(tf.reshape(Y, [-1, CODE_LEN, CHAR_SET_LEN]), 2)
    correction = tf.equal(max_idx_pre, max_idx_label)
    accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))

    # 加载模型
    saver=tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        imgs, labels, txts = get_test_data()
        ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            for i in range(imgs.shape[0]):
                img, label, txt = imgs[i], labels[i], txts[i]
                img = np.reshape(img, [1, INPUT_HEIGHT, INPUT_WIDTH])
                label = np.reshape(label, [1, CODE_LEN * CHAR_SET_LEN])
                pred = sess.run(y, feed_dict={X:img, Y:label, keep_prob: 1.0, is_train: False})
                print("The label is {}, the prediction is {}".format(vec2text(label), vec2text(pred)))
            accuracy_score, preds = sess.run([accuracy,y],feed_dict={X:imgs, Y:labels, keep_prob: 1.0, is_train: False})
            print("After %s step(s), the accuracy on test data is %f"%(global_step, accuracy_score))
        else:
            print("No checkpoint found")


def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()