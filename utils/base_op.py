import tensorflow as tf

def _batch_norm(x, beta, gamma, phase_train, movAve_decay, bn_eps, scope='bn'):
    with tf.variable_scope(scope):
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')  # global normalization
        ema = tf.train.ExponentialMovingAverage(decay=movAve_decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):           #滑动平均值
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, bn_eps)
    return normed

def conv_op(input, var_scope, kh, kw, sh, sw, n_out, is_train, movAve_decay, bn_eps):       #卷积层
    channels = input.shape[-1].value
    with tf.variable_scope(var_scope):
        kernel = tf.get_variable("kernel", dtype=tf.float32, initializer=0.01*tf.random_normal([kh,kw,channels,n_out]))
        bias = tf.get_variable("bias", dtype=tf.float32, initializer=0.1*tf.random_normal(shape=[n_out]))
        layer = tf.nn.conv2d(input, kernel, [1, sh, sw, 1], padding="SAME")
        layer = tf.nn.bias_add(layer, bias)
        layer = _batch_norm(layer, tf.constant(0.0, shape=[n_out]), tf.random_normal(shape=[n_out], mean=1.0, stddev=0.02), is_train, movAve_decay, bn_eps)
        return tf.nn.relu(layer)

def pool_op(input, kh, kw, sh, sw, var_scope):          #池化层
    return tf.nn.max_pool(input, [1, kh, kw, 1], [1, sh, sw, 1], padding="SAME", name=var_scope)

def fc_op(input, var_scope, n_out, regularizer, is_output):        #全连接层
    n_in = input.shape[-1].value
    with tf.variable_scope(var_scope):
        weights = tf.get_variable("weights", dtype=tf.float32, initializer=0.01*tf.random_normal([n_in, n_out]))
        bias = tf.get_variable("bias", dtype=tf.float32, initializer=0.1*tf.random_normal([n_out]))
        if regularizer:
            tf.add_to_collection("losses", regularizer(weights))
        if is_output:
            return tf.add(tf.matmul(input, weights), bias)
        return tf.nn.relu_layer(input, weights, bias, name=var_scope)