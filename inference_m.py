import tensorflow as tf

from utils.CONFIG_M import *
from utils.base_op import *
import numpy as np

def inference(X, seq_len, is_train):
    x = tf.reshape(X, shape=[-1, INPUT_WIDTH, INPUT_HEIGHT, 1])
    conv1 = conv_op(x, "conv1", kh=5, kw=5, sh=1, sw=1, n_out=32, is_train=is_train, movAve_decay=MOVING_AVERAGE_DECAY, bn_eps=BN_EPS)
    pool1 = pool_op(conv1, var_scope="pool1", kh=2, kw=2, sh=2, sw=2)

    # conv2 = conv_op(pool1, "conv2", kh=5, kw=5, sh=1, sw=1, n_out=64, is_train=is_train)
    # pool2 = pool_op(conv2, var_scope="pool2", kh=2, kw=2, sh=2, sw=2)

    conv3 = conv_op(pool1, "conv3", kh=3, kw=3, sh=1, sw=1, n_out=64, is_train=is_train, movAve_decay=MOVING_AVERAGE_DECAY, bn_eps=BN_EPS)
    pool3 = pool_op(conv3, var_scope="pool3", kh=2, kw=2, sh=2, sw=2)

    conv4 = conv_op(pool3, "conv4", kh=3, kw=3, sh=1, sw=1, n_out=64, is_train=is_train, movAve_decay=MOVING_AVERAGE_DECAY, bn_eps=BN_EPS)
    pool4 = pool_op(conv4, var_scope="pool4", kh=2, kw=2, sh=2, sw=2)

    time_steps = pool4.shape[1].value * pool4.shape[2].value
    inputs = tf.reshape(pool4, [-1, time_steps, pool4.shape[3].value])   #结果　[batch_size, 256, 64],　输入到lstm

    #构造sequence_length
    # seq_len = np.ones(inputs.shape[0]) * inputs.shape[1]
    cell = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN, state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)     #第二个输出是rnn的最后一个state, outputs [batch_size, time_steps, hidden_state]

    outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])     #[batch_size * time_steps, num_hidden]
    W = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_CLASSES], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]), name="b")

    logits = tf.matmul(outputs, W) + b      #[batch_size * time_steps, num_classes]

    logits = tf.reshape(logits, [-1, 256, NUM_CLASSES])  #[batch_num, time_steps, num_classes]

    logits = tf.transpose(logits, (1, 0, 2))  #[time_steps, batch_size, num_classes]作为ctc_loss的输入

    return tf.nn.softmax(logits)


def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0

    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    print("Accuracy:", true_numer * 1.0 / len(original_list))

def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = CHAR_SET[spars_tensor[1][m]]
        decoded.append(str)
    # Replacing blank label to none
    #str_decoded = str_decoded.replace(chr(ord('9') + 1), '')
    # Replacing space label to space
    #str_decoded = str_decoded.replace(chr(ord('0') - 1), ' ')
    # print("ffffffff", str_decoded)
    return decoded

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    #print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        #print("index = ", index)
        result.append(decode_a_seq(index, sparse_tensor))
        #print(result)
    return result
