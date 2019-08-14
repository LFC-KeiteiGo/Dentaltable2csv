from utils.util_outputs import *
from utils.data_house import DataHouse

import os
import tensorflow as tf


# Read prediction
dws = DataHouse()
dws.pred_data_path = './LocalData/output_cellparsed/'
dws.pred_data_prep()

# Predictions by saved model
tf.reset_default_graph()
sess = tf.Session()
prediction_main = tf_predict(sess, dws.datam_pred, './model/model1/model1.ckpt', 128)
sess.close()

tf.reset_default_graph()
sess = tf.Session()
prediction_sub = tf_predict(sess, dws.datas_pred, './model/model2/model2.ckpt', 128)
sess.close()

# data to CSV
table_prediction = standard_table('LocalData/output_cellparsed', dws, prediction_main, prediction_sub)
table_prediction.to_csv('outputs/generallist.csv')

# simple visualization
text_creator = TextFormatTable(prediction_main, prediction_sub, dws.listm_pred, dws.lists_pred)
text_creator.show('PNum0001')

# Image merged representations
pnums = os.listdir('LocalData/output_cellparsed')
img_creator = ImageFormatTable('LocalData/output_cellparsed', dws, prediction_main, prediction_sub)
for pnum in pnums:
    img = img_creator.do(pnum)
    cv2.imwrite('outputs/'+pnum+'.jpg', img)
