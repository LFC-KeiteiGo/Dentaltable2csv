import tensorflow as tf
import numpy as np

from codes.DataHouse import DataHouse
from codes.ConvSetting import Model
from tensorflow.examples.tutorials.mnist import input_data

dws = DataHouse()

dws.mnist = input_data.read_data_sets("./MNIST-data", one_hot=True)
dws.local_marked_path = './LocalData/Annotated/done/'
dws.external_data_path = './numbers/'  # Credits to https://github.com/kensanata/numbers
dws.augment_data_path = './AugmentorData/'
dws.pred_data_path = './LocalData/output_cellparsed/'

dws.train_data_prep(aug_count=8000, ext_count=8000)  # Tune number of traindata
dws.pred_data_prep()

#######
# Train Caries & Alveolar
#######
dws.reset()
# Training Parameters
model1 = Model(dws, input_size=[28, 28], class_num=8, learning_rate=0.0007,
               steps=100000, batch_size=128, drop_out=0.8)
# Set weights
model1.create_weights(wc1=([4, 4, 1, 32], 1.0), wc2=([4, 4, 32, 32], 1/32),
                      wc3=([3, 3, 32, 64], 1/32), wc4=([3, 3, 64, 64], 1/32),
                      wc5=([3, 3, 64, 128], 1/128), wc6=([2, 2, 128, 128], 1/128),
                      wc7=([2, 2, 128, 128], 1/128), wd1=([4 * 4 * 128, 512], 1.0))
# Set biases
model1.create_biases(bc1=([32], 1.0), bc2=([32], 1.0), bc3=([64], 1.0), bc4=([64], 1.0),
                     bc5=([128], 1.0), bc6=([128], 1.0), bc7=([128], 1.0), bd1=([512], 1.0))
# Create Network
model1.building_network(Conv1=('wc1', 'bc1'), Conv2=('wc2', 'bc2'), Maxpool1=2,
                        Conv3=('wc3', 'bc3'), Conv4=('wc4', 'bc4'), Maxpool2=2,
                        Conv5=('wc5', 'bc5'), Conv6=('wc6', 'bc6'), Maxpool3=2,
                        Conv7=('wc7', 'bc7'), FCL1=('wd1', 'bd1'))
sess = tf.Session()
model1.default_preproc()
model1.train(sess, 'main')
model1.save_model(sess, './model/model1/model1.ckpt')
prediction_main = model1.predict(sess, 'main')
sess.close()


######
# Train ApicalLesion & Others
######
dws.reset()
# Create place holders for data
model2 = Model(dws, input_size=[45, 90], class_num=5, learning_rate=0.001,
               steps=120000, batch_size=128, drop_out=0.75)
# Set weights
model2.create_weights(wc1=([5, 5, 1, 32], 1/3), wc2=([5, 5, 32, 32], 1/32),
                      wd1=([10 * 5 * 32, 256], 1.0))
# Set biases
model2.create_biases(bc1=([32], 1.0), bc2=([32], 1.0), bd1=([256], 1.0))

# Create Network
model2.building_network(Conv1=('wc1', 'bc1'), Maxpool1=3,
                        Conv2=('wc2', 'bc2'), Maxpool2=3,
                        FCL1=('wd1', 'bd1'))
sess = tf.Session()
model2.default_preproc()
model2.train(sess, 'sub')
model2.save_model(sess, './model/model2/model2.ckpt')
prediction_sub = model2.predict(sess, 'sub')
sess.close()


