import tensorflow as tf
import numpy as np

from codes.DataHouse import DataHouse
from codes.ConvSetting import init_var, conv_net_main, conv_net_sub, conv_net_8

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

dws = DataHouse()

dws.mnist = mnist
dws.local_marked_path = 'C:/Users/neilc/Projects/Dental_Panorama/image_dataformat/Annotated/done/'
dws.external_data_path = 'C:/Users/neilc/GitHub/numbers/'
dws.augment_data_path = 'C:/Users/neilc/Projects/Dental_Panorama/AugmentorData/'
dws.pred_data_path = 'C:/Users/neilc/Projects/Dental_Panorama/image_dataformat/output_cellparsed/'

dws.train_data_prep(aug_count=8000, ext_count=8000)  # Tune Traindata
dws.pred_data_prep()

#######
# Train Caries & Alveolar
#######
dws.reset()

# Training Parameters
learning_rate = 0.0007
num_steps = 25000
batch_size = 128
display_step = 100
resize = [28, 28]

# Network Parameters
num_input = 784  # MNIST mained data input (img shape: 28*28)
num_classes = 8  # Total classes (blank+0-7 digits)
dropout = 0.8  # Dropout, probability to keep units

# Create place holders for data
X, Y, Pred, keep_prob = init_var(num_input, num_classes)

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([4, 4, 1, 32]), name='wc1'),
    'wc2': tf.Variable(tf.random_normal([4, 4, 32, 32], stddev=1 / 32), name='wc2'),
    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=1 / 32), name='wc3'),
    'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=1 / 32), name='wc4'),
    'wc5': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=1 / 128), name='wc5'),
    'wc6': tf.Variable(tf.random_normal([2, 2, 128, 128], stddev=1 / 128), name='wc6'),
    'wc7': tf.Variable(tf.random_normal([2, 2, 128, 128], stddev=1 / 128), name='wc7'),
    'wd1': tf.Variable(tf.random_normal([4 * 4 * 128, 512]), name='wd1'),
    'out': tf.Variable(tf.random_normal([512, num_classes]), name='out')
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name='bc1'),
    'bc2': tf.Variable(tf.random_normal([32]), name='bc2'),
    'bc3': tf.Variable(tf.random_normal([64]), name='bc3'),
    'bc4': tf.Variable(tf.random_normal([64]), name='bc4'),
    'bc5': tf.Variable(tf.random_normal([128]), name='bc5'),
    'bc6': tf.Variable(tf.random_normal([128]), name='bc6'),
    'bc7': tf.Variable(tf.random_normal([128]), name='bc7'),
    'bd1': tf.Variable(tf.random_normal([512]), name='bd1'),
    'out': tf.Variable(tf.random_normal([num_classes]), name='out')
}

# Construct model
logits = conv_net_main(X, weights, biases, keep_prob, True, resize)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                         logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Define accuracy
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Define External prediction
predict_model = conv_net_main(Pred, weights, biases, keep_prob, False, resize)
pred = tf.nn.softmax(predict_model)

# Initialize variables
init = tf.global_variables_initializer()

# Start training
sess = tf.Session()

# Set iterations
dws.set_iter(num_steps)

# Run the initializer
sess.run(init)

for step in range(1, num_steps + 1):
    batch_x, batch_y = dws.serve_dish_train('main', batch_size)
    # Run optimization op (backprop)
    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                             Y: batch_y,
                                                             keep_prob: 1.0})
        print("Step " + str(step) + ", Minibatch Loss= " +
              "{:.4f}".format(loss) + ", Training Accuracy= " +
              "{:.3f}".format(acc))

print("Optimization Finished!")

first_pred = True
prediction_main = None  # Pre-declaration of variable
logits_main = None  # Pre-declaration of variable
for idx_start in range(0, len(dws.listm_pred), batch_size):
    data_seg = dws.datam_pred[idx_start:min((idx_start + batch_size), len(dws.listm_pred))]
    pred_seg = sess.run(pred, feed_dict={Pred: data_seg, keep_prob: 1.0})
    logits_seg = sess.run(predict_model, feed_dict={Pred: data_seg, keep_prob: 1.0})
    if first_pred:
        prediction_main = pred_seg
        logits_main = logits_seg
        first_pred = False
        continue
    prediction_main = np.vstack((prediction_main, pred_seg))
    logits_main = np.vstack((logits_main, logits_seg))

sess.close()

######
# Train ApicalLesion & Others
######
dws.reset()

# Training Parameters
learning_rate = 0.001
num_steps = 25000
batch_size = 128
display_step = 100
resize = [45, 90]

# Network Parameters
num_input = 4050  # Data input (img shape: 45*90)
num_classes = 5  # Total classes
dropout = 0.7  # Dropout, probability to keep units

# Create place holders for data
X, Y, Pred, keep_prob = init_var(num_input, num_classes)

# Store layers weight & bias
weights_sub = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=1 / 3), name='wc1'),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 32], stddev=1 / 32), name='wc2'),
    'wd1': tf.Variable(tf.random_normal([10 * 5 * 32, 256]), name='wd1'),
    'out': tf.Variable(tf.random_normal([256, num_classes]), name='out')
}

biases_sub = {
    'bc1': tf.Variable(tf.random_normal([32]), name='bc1'),
    'bc2': tf.Variable(tf.random_normal([32]), name='bc2'),
    'bd1': tf.Variable(tf.random_normal([256]), name='bd1'),
    'out': tf.Variable(tf.random_normal([num_classes]), name='out')
}

# Construct model
logits = conv_net_sub(X, weights_sub, biases_sub, keep_prob, True, resize)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Define accuracy
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Define External prediction
predict_model = conv_net_sub(Pred, weights_sub, biases_sub, keep_prob, False, resize)
pred = tf.nn.softmax(predict_model)

# Initialize variables
init = tf.global_variables_initializer()

# Start training
sess = tf.Session()

# Set iterations
dws.set_iter(num_steps)

# Run the initializer
sess.run(init)

for step in range(1, num_steps + 1):
    batch_x, batch_y = dws.serve_dish_train('sub', batch_size)
    # Run optimization op (backprop)
    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                             Y: batch_y,
                                                             keep_prob: 1.0})
        print("Step " + str(step) + ", Minibatch Loss= " +
              "{:.4f}".format(loss) + ", Training Accuracy= " +
              "{:.3f}".format(acc))

print("Optimization Finished!")

prediction_sub = None
first_pred = True
for idx_start in range(0, len(dws.lists_pred), batch_size):
    data_seg = dws.datas_pred[idx_start:min((idx_start + batch_size), len(dws.lists_pred))]
    pred_seg = sess.run(pred, feed_dict={Pred: data_seg, keep_prob: 1.0})
    if first_pred:
        prediction_sub = pred_seg
        first_pred = False
        continue
    prediction_sub = np.vstack((prediction_sub, pred_seg))

sess.close()

