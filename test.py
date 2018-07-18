import numpy as np
import tensorflow as tf

batch_size = 4
sequence_max_len = 5
num_classes = 3

y_pred = tf.placeholder(tf.float32, shape=(batch_size, sequence_max_len, num_classes))
y_pred_transposed = tf.transpose(y_pred,
                                 perm=[1, 0, 2])  # TF expects dimensions [max_time, batch_size, num_classes]
logits = tf.log(y_pred_transposed)
sequence_lengths = tf.to_int32(tf.fill([batch_size], sequence_max_len))
decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(logits,
                                                           sequence_length=sequence_lengths,
                                                           beam_width=3,
                                                           merge_repeated=False, top_paths=1)

decoded = decoded[0]
decoded_paths = tf.sparse_tensor_to_dense(decoded)  # Shape: [batch_size, max_sequence_len]

with tf.Session() as session:
    tf.global_variables_initializer().run()

    softmax_outputs = np.array([[[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]],
                                [[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]],
                                [[0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]],
                                [[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]]])

    decoded_paths = session.run([decoded_paths], feed_dict = {y_pred: softmax_outputs})
    print(decoded_paths)

#
# sf = [[[0.33571429, 0.48571429 ,0.17857143],[0.38728324, 0.47976879, 0.13294798], [0.56441718 ,0.34969325, 0.08588957], [0.125,0.39130435,0.48369565],[0.3,0.64285714, 0.05714286]],
#
#  [[0.2516129 , 0.43870968, 0.30967742]  ,
#   [0.1372549  ,0.8627451 , 0.]  ,
#   [0.55147059 ,0.40441176, 0.04411765]
#   [0.15447154 ,0.48780488, 0.35772358]
#   [0.33510638 ,0.36702128 ,0.29787234]]  ,
#
#  [[0.18181818, 0.41666667, 0.40151515]
#   [0.3836478 , 0.40251572 ,0.21383648]
#   [0.2705314 , 0.352657  , 0.37681159]
#   [0.74509804 ,0.07843137 ,0.17647059]
#   [0.34387352 ,0.39130435 ,0.26482213]]  ,
#
#  [[0.3546798 , 0.408867 ,  0.2364532 ]
#   [0.01234568, 0.79012346, 0.19753086]
#   [0.18452381 ,0.55357143, 0.26190476]
#   [0.49462366, 0.38172043 ,0.12365591]
#   [0.09174312, 0.32110092, 0.58715596]]]
#