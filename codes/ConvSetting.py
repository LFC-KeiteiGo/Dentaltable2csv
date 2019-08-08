import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, datahouse, input_size, class_num, learning_rate,
                 steps, batch_size, drop_out, display_step=100):
        self.dws = datahouse
        self.size = input_size
        self.input_num = input_size[0]*input_size[1]
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.steps_num = steps
        self.drop_out = drop_out
        self.batch_size = batch_size
        self.display_step = display_step

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.input_num], name='X')
        self.y = tf.compat.v1.placeholder(tf.float32, [None, self.class_num], name='Y')
        self.val_x = tf.compat.v1.placeholder(tf.float32, [None, self.input_num], name='Val_X')
        self.val_y = tf.compat.v1.placeholder(tf.float32, [None, self.class_num], name='Val_Y')
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='KP')

        self.weights = {}
        self.biases = {}
        self.network_train = None
        self.network_valid = None
        self.trainer = None
        self.loss = None
        self.accuracy = None

        self.shape = None

    @staticmethod
    def conv2d(x, w, b, strides=1, padding='SAME'):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    @staticmethod
    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool2d(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    @staticmethod
    def fullyconnect(x, w, b):
        x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
        x = tf.add(tf.matmul(x, w), b)
        return tf.nn.relu(x)

    @staticmethod
    def weight_variable(name, shape, stdev):
        initial = tf.truncated_normal(shape, stddev=stdev)
        return tf.Variable(initial, name=name)

    def _add_layers(self, inputs, layer_type, param_names):
        if 'Conv' in layer_type:
            return self.conv2d(inputs, self.weights[param_names[0]], self.biases[param_names[1]])
        if 'Maxpool' in layer_type:
            return self.maxpool2d(inputs, param_names)
        if 'FCL' in layer_type:
            return self.fullyconnect(inputs, self.weights[param_names[0]], self.biases[param_names[1]])

    # Model.create_weights(wc1=([kenel_x, kernek_y, inpu_num, filter_count], stdev),
    #                      wc2=([kenel_x, kernek_y, inpu_num, filter_count], stdev))
    def create_weights(self, **kwargs):
        for name, stats in kwargs.items():
            self.shape, stdev = stats
            self.weights[name] = self.weight_variable(name, self.shape, stdev)
        self.weights['out'] = self.weight_variable('out', [self.shape[-1], self.class_num], 1.0)

    # Model.create_biases(bc1=([filter_count], stdev),
    #                     bc2=([filter_count], stdev))
    def create_biases(self, **kwargs):
        for name, stats in kwargs.items():
            self.shape, stdev = stats
            self.biases[name] = self.weight_variable(name, self.shape, stdev)
        self.biases['out'] = self.weight_variable('out', [self.class_num], 1.0)

    # Miracle Function **
    # Model.building_network(Conv1=('wc1', 'bc1'), Maxpool1=3, FCL1=('wd1', 'bd1'))
    #                        1st layer(attr) > 2nd layer(attr) > 3rd layer(attr)
    def building_network(self, **kwargs):
        size_x, size_y = self.size
        x = tf.reshape(self.x, shape=[-1, size_y, size_x, 1])
        val = tf.reshape(self.val_x, shape=[-1, size_y, size_x, 1])

        value_train = None
        value_valid = None
        net_first = True

        for layer_type, param_names in kwargs.items():
            if net_first:
                value_train = self._add_layers(x, layer_type, param_names)
                value_valid = self._add_layers(val, layer_type, param_names)
                net_first = False
            value_train = self._add_layers(value_train, layer_type, param_names)
            value_valid = self._add_layers(value_valid, layer_type, param_names)
        value_train = tf.nn.dropout(value_train, self.keep_prob)
        self.network_train = tf.add(tf.matmul(value_train, self.weights['out']), self.biases['out'])
        self.network_valid = tf.add(tf.matmul(value_valid, self.weights['out']), self.biases['out'])

    def _loss_func(self):
        t = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.network_train, labels=self.y)
        self.loss = tf.reduce_mean(t)

    def _adam_optimize_operation(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.trainer = optimizer.minimize(self.loss)

    def _efficacy_check(self):
        prediction = tf.nn.softmax(self.network_train)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def default_preproc(self):
        self._loss_func()
        self._adam_optimize_operation()
        self._efficacy_check()

    def train(self, sess, ms):
        self.dws.set_iter(self.steps_num)
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(1, self.steps_num + 1):
            batch_x, batch_y = self.dws.serve_dish_train(ms, self.batch_size)
            # Run optimization op (backprop)
            sess.run(self.trainer, feed_dict={self.x: batch_x, self.y: batch_y,
                                              self.keep_prob: self.drop_out})
            if step % self.display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([self.loss, self.accuracy], feed_dict={self.x: batch_x,
                                                                            self.y: batch_y,
                                                                            self.keep_prob: 1.0})
                print("Step {}, BatchLoss= {:.4f}, TrainAccuracy= {:.3f}".format(str(step), loss, acc))
        print("Optimization Finished!")

    def predict(self, sess, ms):
        predict = tf.nn.softmax(self.network_valid)
        f_lists = self.dws.listm_pred if ms is 'main' else self.dws.lists_pred
        p_datas = self.dws.datam_pred if ms is 'main' else self.dws.datas_pred
        first_pred = True
        predictions = None  # Pre-declaration of variable

        for idx_start in range(0, len(f_lists), self.batch_size):
            data_seg = p_datas[idx_start:min((idx_start + self.batch_size), len(f_lists))]
            pred_seg = sess.run(predict, feed_dict={self.val_x: data_seg, self.keep_prob: 1.0})
            if first_pred:
                predictions = pred_seg
                first_pred = False
                continue
            predictions = np.vstack((predictions, pred_seg))

        return predictions
