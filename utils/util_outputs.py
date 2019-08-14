import pandas as pd
import numpy as np
import re
import os
import cv2
import tensorflow as tf
from utils.data_house import concatstack


def extract_file(pnum, pred_class, pred_file):
    idx = [i for i, x in enumerate(pred_file) if pnum in x]
    class_chunk = [list(pred_class[i]) for i in idx]
    file_chunk = [pred_file[i] for i in idx]
    return file_chunk, class_chunk


def tf_predict(sess, data, ckpt_path, batch_size):
    saver = tf.train.import_meta_graph(ckpt_path+'.meta')
    saver.restore(sess, ckpt_path)
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name('Val_X:0')
    keep_prob = graph.get_tensor_by_name('KP:0')
    predict_func = graph.get_tensor_by_name('predict:0')

    first_pred = True
    predictions = None  # Pre-declaration of variable

    for idx_start in range(0, len(data), batch_size):
        data_seg = data[idx_start:min((idx_start + batch_size), len(data))]
        pred_seg = sess.run(predict_func, feed_dict={x: data_seg, keep_prob: 1.0})
        if first_pred:
            predictions = pred_seg
            first_pred = False
            continue
        predictions = np.vstack((predictions, pred_seg))
    return predictions


# Usage:
# output_prediction('PNum0300', prediction_main, prediction_sub, dws.pred_listm, dws.pred_lists)
# showtext = TextFormatTable(class_main, class_sub, filelist_main, filelist_sub)
# showtext.show('PNum0001')
class TextFormatTable:
    def __init__(self, class_main, class_sub, filelist_main, filelist_sub):
        self._class_main = class_main
        self._class_sub = class_sub
        self._filelist_main = filelist_main
        self._filelist_sub = filelist_sub

    @staticmethod
    def _init_docform():
        template = pd.DataFrame(np.zeros(shape=(9, 17)))
        template.iloc[:, 0] = ['その他の病変', '根尖病巣', '歯槽頂', '齲蝕', '', '齲蝕', '歯槽頂', '根尖病巣', 'その他の病変']
        template.iloc[4, :] = ['', 'R8', 'R7', 'R6', 'R5', 'R4', 'R3', 'R2', 'R1',
                               'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']
        return template

    @staticmethod
    def _fill_prediction(form, pnum, classes, filelist, ms):
        notations = None
        sub_files, sub_classes = extract_file(pnum, classes, filelist)
        y_idx1 = {'Others': [0, 8], 'ApicalLesion': [1, 7], 'Alveolar': [2, 6], 'Caries': [3, 5]}
        y_idx2 = {'U': list(range(0, 5)), 'D': list(range(5, 9))}

        if ms is 'main':
            notations = ['-', '1', '2', '3', '4', '5', '6', '7']
        elif ms is 'sub':
            notations = ['-', '/', '有', 'インプ', '他']

        for i, file in enumerate(sub_files):
            name_chunk = re.split('_|\.', file)
            key_ud = name_chunk[1][1]
            key_row = name_chunk[1][0] + name_chunk[1][2]
            key_symptom = name_chunk[2]

            idx_y = [i for i in y_idx2[key_ud] if i in y_idx1[key_symptom]][0]
            idx_x = form.iloc[4, :].tolist().index(key_row)

            if max(sub_classes[i]) > 0.7:
                idx_class = sub_classes[i].index(max(sub_classes[i]))
                notation = notations[idx_class]
            elif max(sub_classes[i]) > 0.4:
                idx_class = sub_classes[i].index(max(sub_classes[i]))
                notation = notations[idx_class] + '?'
            else:
                idx_class = sub_classes[i].index(max(sub_classes[i]))
                notation = notations[idx_class] + '??'

            form.iloc[idx_y, idx_x] = notation
        return form

    def show(self, patient_num):
        print('Printing table...')
        form = self._init_docform()
        form = self._fill_prediction(form, patient_num, self._class_main, self._filelist_main, 'main')
        form = self._fill_prediction(form, patient_num, self._class_sub, self._filelist_sub, 'sub')
        for i in range(1, 17):
            if '/' in form.iloc[[0, 1], i].tolist():
                form.iloc[[0, 1, 2, 3], i] = '/'
            if '/' in form.iloc[[7, 8], i].tolist():
                form.iloc[[5, 6, 7, 8], i] = '/'
        print(form)


# Usage:
# img_creator = ImageFormatTable(path, data_house, class_main, class_sub)
# img = img_creator.do(pnum)
class ImageFormatTable:
    def __init__(self, path, data_house, class_main, class_sub):
        print('Creating merged image data...')
        self.path = path
        self.dws = data_house
        self._class_main = class_main
        self._class_sub = class_sub

        self._imgs = None
        self._fnames = None

    @staticmethod
    def _add_pred2_img(img, pred_class, ms):
        notations = []
        if ms is 'main':
            notations = [' ', '1', '2', '3', '4', '5', '6', '7']
        elif ms is 'sub':
            notations = [' ', '/', 'H', 'I', 'E']
        elif ms is 'slash':
            img = cv2.putText(img, '/', (0, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, (0, 0, 0))
            return img
        label = notations[pred_class.index(max(pred_class))]
        img = cv2.putText(img, label, (0, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, (0, 0, 0))
        return img

    def _load_singlepatient(self, pnum):
        self._fnames = os.listdir(self.path + '/'+pnum)
        self._imgs = [cv2.imread('{}/{}'.format(self.path+'/'+pnum, file)) for file in self._fnames]

    def do(self, pnum):
        self._load_singlepatient(pnum)
        img = self.create_tableimage()
        return img

    def create_tableimage(self):
        imgs, fnames, dws = self._imgs, self._fnames, self.dws
        prediction_main = self._class_main
        prediction_sub = self._class_sub
        # Create Top
        top_tooth = ['RU8', 'RU7', 'RU6', 'RU5', 'RU4', 'RU3', 'RU2', 'RU1',
                     'LU1', 'LU2', 'LU3', 'LU4', 'LU5', 'LU6', 'LU7', 'LU8']
        top_situ = ['Others', 'ApicalLesion', 'Alveolar', 'Caries']

        first_col = True
        first_row = True
        its_slash = False

        col_img = None
        big_topimg = None
        new_img = None

        for teeth in top_tooth:
            # Return [File Index, File Name]
            file_sub = [[i, x] for i, x in enumerate(fnames) if teeth in x]
            for situation in top_situ:

                try:
                    file = [x for x in file_sub if situation in x[1]][0]
                    if its_slash:
                        new_img = self._add_pred2_img(imgs[file[0]], [], ms='slash')
                    elif situation in ['Others', 'ApicalLesion']:
                        idx = dws.lists_pred.index(file[1])
                        pred = list(prediction_sub[idx])
                        new_img = self._add_pred2_img(imgs[file[0]], pred, 'sub')
                        if pred.index(max(pred)) is 1:
                            its_slash = True
                    elif situation in ['Alveolar', 'Caries']:
                        idx = dws.listm_pred.index(file[1])
                        pred = list(prediction_main[idx])
                        new_img = self._add_pred2_img(imgs[file[0]], pred, 'main')
                except Exception as e:
                    print('Raise warnings at {} because of:{}'.format(teeth, e))
                    new_img = np.ones([47, 92, 3], dtype=np.int8) * 255
                if first_col:
                    col_img = new_img
                    first_col = False
                    continue
                col_img = np.vstack((col_img, new_img))

            its_slash = False
            if first_row:
                big_topimg = col_img
                first_row = False
                first_col = True
                continue
            big_topimg = np.hstack((big_topimg, col_img))
            first_col = True

        # Create Bottom
        bot_tooth = ['RD8', 'RD7', 'RD6', 'RD5', 'RD4', 'RD3', 'RD2', 'RD1',
                     'LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'LD7', 'LD8']
        bot_situ = ['Others', 'ApicalLesion', 'Alveolar', 'Caries']
        first_col = True
        first_row = True
        its_slash = False

        col_img = None
        big_botimg = None
        new_img = None

        for teeth in bot_tooth:
            # Return [File Index, File Name]
            file_sub = [[i, x] for i, x in enumerate(fnames) if teeth in x]
            for situation in bot_situ:
                try:
                    file = [x for x in file_sub if situation in x[1]][0]
                    if its_slash:
                        new_img = self._add_pred2_img(imgs[file[0]], [], ms='slash')
                    elif situation in ['Others', 'ApicalLesion']:
                        idx = dws.lists_pred.index(file[1])
                        pred = list(prediction_sub[idx])
                        new_img = self._add_pred2_img(imgs[file[0]], pred, 'sub')
                        if pred.index(max(pred)) is 1:
                            its_slash = True
                    elif situation in ['Alveolar', 'Caries']:
                        idx = dws.listm_pred.index(file[1])
                        pred = list(prediction_main[idx])
                        new_img = self._add_pred2_img(imgs[file[0]], pred, 'main')
                except Exception as e:
                    print('Raise warnings at {} because of:{}'.format(teeth, e))
                    new_img = np.ones([47, 92, 3], dtype=np.int8) * 255

                if first_col:
                    col_img = new_img
                    first_col = False
                    continue
                col_img = np.vstack((new_img, col_img))

            its_slash = False
            if first_row:
                big_botimg = col_img
                first_row = False
                first_col = True
                continue
            big_botimg = np.hstack((big_botimg, col_img))
            first_col = True

        # big_img = np.vstack((big_topimg, big_botimg))
        # Create Middle
        tooth = ['R8', 'R7', 'R6', 'R5', 'R4', 'R3', 'R2', 'R1',
                 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']
        first = True
        ref_hori = []
        for teeth in tooth:
            empty = np.ones([45, 90, 3], dtype=np.int8)*255
            img = cv2.putText(empty, teeth, (20, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 4)
            img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, (0, 0, 0))
            if first:
                ref_hori = img
                first = False
            else:
                ref_hori = np.hstack((ref_hori, img))

        # Create Left
        situations = ['Others', 'Apical', 'Perio', 'Caries', '',
                      'Caries', 'Perio', 'Apical', 'Others']
        first = True
        ref_vert = []
        for situ in situations:
            empty = np.ones([45, 120, 3], dtype=np.int8)*255
            img = cv2.putText(empty, situ, (20, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, (0, 0, 0))
            if first:
                ref_vert = img
                first = False
            else:
                ref_vert = np.vstack((ref_vert, img))

        big_img = concatstack((big_topimg, ref_hori, big_botimg))
        big_img = np.hstack((ref_vert, big_img))

        return big_img


def standard_table(path, data_house, class_main, class_sub):
    print('Converting data to csv format...')
    plist = os.listdir(path)
    plist = [x for x in plist if 'PNum' in x]

    teeth = ['LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'LD7', 'LD8', 'LU1', 'LU2', 'LU3', 'LU4', 'LU5', 'LU6', 'LU7',
             'LU8',
             'RD1', 'RD2', 'RD3', 'RD4', 'RD5', 'RD6', 'RD7', 'RD8', 'RU1', 'RU2', 'RU3', 'RU4', 'RU5', 'RU6', 'RU7',
             'RU8']
    notations_main = ['0', '1', '2', '3', '4', '5', '6', '7']
    notations_apical = ['0', '0', '1', '0', '0']

    data = ['No', 'Pos', 'Exist', 'Caries', 'PeriodontalDisease', 'ApicalLesion']
    template = pd.DataFrame(columns=data)

    for patient in plist:
        pnum = int(patient[4:])
        x_m, y_m = extract_file(patient, class_main, data_house.listm_pred)
        x_s, y_s = extract_file(patient, class_sub, data_house.lists_pred)
        file_names = x_m + x_s
        classes = y_m + y_s

        for tooth in teeth:
            new_row = pd.DataFrame({'No': [pnum], 'Pos': tooth, 'Exist': [1],
                                    'Caries': [0], 'PeriodontalDisease': [0], 'ApicalLesion': [0]})
            files_tooth = [x for x in file_names if tooth in x]
            files_sub = [x for y in ['Others', 'ApicalLesion'] for x in files_tooth if y in x]
            classes_sub = np.array([np.array(classes[file_names.index(x)]) for x in files_sub])

            # Check Exist
            if sum(classes_sub)[1] > 0.6:
                new_row['Exist'] = 0
                template = pd.concat([template, new_row], ignore_index=True)
                continue

            # Check Caries
            status_caries = [classes[file_names.index(x)] for x in files_tooth if 'Caries' in x][0]
            new_row['Caries'] = notations_main[status_caries.index(max(status_caries))]

            # Check Periodontal
            status_perio = [classes[file_names.index(x)] for x in files_tooth if 'Alveolar' in x][0]
            new_row['PeriodontalDisease'] = notations_main[status_perio.index(max(status_perio))]

            # Check Apical
            status_apical = [classes[file_names.index(x)] for x in files_tooth if 'ApicalLesion' in x][0]
            new_row['ApicalLesion'] = notations_apical[status_apical.index(max(status_apical))]

            template = pd.concat([template, new_row], ignore_index=True)
    return template
