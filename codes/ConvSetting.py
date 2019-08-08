import tensorflow as tf


def conv2d(x, w, b, strides=1, padding='SAME'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def init_var(num_input, num_classes):
    x = tf.compat.v1.placeholder(tf.float32, [None, num_input], name='X')
    y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name='Y')
    pred = tf.compat.v1.placeholder(tf.float32, [None, num_input], name='Pred')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='KP')
    return x, y, pred, keep_prob


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool2d(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                            padding='SAME')


# Create model
def conv_net_main(x, weights, biases, rate, train, size):
    size_x, size_y = size
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, size_y, size_x, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2)

    # Convolution Layer
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'])
    # Max Pooling (down-sampling)
    conv6 = maxpool2d(conv6, k=2)

    # Convolution Layer
    conv7 = conv2d(conv6, weights['wc7'], biases['bc7'])

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv7, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    if train:
        fc1 = tf.nn.dropout(fc1, rate)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Create model
def conv_net_sub(x, weights, biases, dropout, train, size):
    size_x, size_y = size
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, size_y, size_x, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=3)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=3)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    if train:
        fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out