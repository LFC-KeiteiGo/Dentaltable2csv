import pandas as pd
import numpy as np
import re
import os
import cv2

def init_docform():
    template = pd.DataFrame(np.zeros(shape=(9, 17)))
    template.iloc[:, 0] = ['その他の病変', '根尖病巣', '歯槽頂', '齲蝕', '', '齲蝕', '歯槽頂', '根尖病巣', 'その他の病変']
    template.iloc[4, :] = ['', 'R8', 'R7', 'R6', 'R5', 'R4', 'R3', 'R2', 'R1',
                           'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']
    return template


def fill_prediction(form, pnum, classes, filelist, ms):
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


def output_prediction(PNum, class_main, class_sub, filelist_main, filelist_sub):
    form = init_docform()
    form = fill_prediction(form, PNum, class_main, filelist_main, 'main')
    form = fill_prediction(form, PNum, class_sub, filelist_sub, 'sub')
    for i in range(1,17):
        if '/' in form.iloc[[0, 1], i].tolist():
            form.iloc[[0, 1, 2, 3], i] = '/'
        if '/' in form.iloc[[7, 8], i].tolist():
            form.iloc[[5, 6, 7, 8], i] = '/'
    return form


def extract_file(pnum, pred_class, pred_file):
    idx = [i for i, x in enumerate(pred_file) if pnum in x]
    class_chunk = [list(pred_class[i]) for i in idx]
    file_chunk = [pred_file[i] for i in idx]
    return file_chunk, class_chunk


def load_singlepatient(path, pnum):
    fnames = os.listdir(path+'/'+pnum)
    imgs = [cv2.imread('{}/{}'.format(path+'/'+pnum, file)) for file in fnames]
    return imgs, fnames


def add_pred2_img(img, pred_class, ms):
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


def create_tableimage(imgs, fnames, dws, p_main, p_sub):
    prediction_main = p_main
    prediction_sub = p_sub
    # Create Top
    Top_tooth = ['RU8', 'RU7', 'RU6', 'RU5', 'RU4', 'RU3', 'RU2', 'RU1',
                 'LU1', 'LU2', 'LU3', 'LU4', 'LU5', 'LU6', 'LU7', 'LU8']
    Top_situ = ['Others', 'ApicalLesion', 'Alveolar', 'Caries']

    first_col = True
    first_row = True
    its_slash = True

    col_img = None
    big_topimg = None
    new_img = None

    #todo: test here using real data, load_singlepatient function tested
    for teeth in Top_tooth:
        # Return [File Index, File Name]
        file_sub = [[i, x] for i, x in enumerate(fnames) if teeth in x]
        for situation in Top_situ:

            try:
                file = [x for x in file_sub if situation in x[1]][0]
                if its_slash:
                    new_img = add_pred2_img(imgs[file[0]], [], ms='slash')
                elif situation in ['Others', 'ApicalLesion']:
                    idx = dws.lists_pred.index(file[1])
                    pred = list(prediction_sub[idx])
                    new_img = add_pred2_img(imgs[file[0]], pred, 'sub')
                    if pred.index(max(pred)) is 1:
                        its_slash = True
                elif situation in ['Alveolar', 'Caries']:
                    idx = dws.listm_pred.index(file[1])
                    pred = list(prediction_main[idx])
                    new_img = add_pred2_img(imgs[file[0]], pred, 'main')
            except Exception as e:
                print(e)
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
    Bot_tooth = ['RD8', 'RD7', 'RD6', 'RD5', 'RD4', 'RD3', 'RD2', 'RD1',
                 'LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'LD7', 'LD8']
    Bot_situ = ['Others', 'ApicalLesion', 'Alveolar', 'Caries']
    first_col = True
    first_row = True
    its_slash = True

    col_img = None
    big_botimg = None
    new_img = None

    for teeth in Bot_tooth:
        # Return [File Index, File Name]
        file_sub = [[i, x] for i, x in enumerate(fnames) if teeth in x]
        for situation in Bot_situ:
            try:
                file = [x for x in file_sub if situation in x[1]][0]
                if its_slash:
                    new_img = add_pred2_img(imgs[file[0]], [], ms='slash')
                elif situation in ['Others', 'ApicalLesion']:
                    idx = dws.lists_pred.index(file[1])
                    pred = list(prediction_sub[idx])
                    new_img = add_pred2_img(imgs[file[0]], pred, 'sub')
                    if pred.index(max(pred)) is 1:
                        its_slash = True
                elif situation in ['Alveolar', 'Caries']:
                    idx = dws.listm_pred.index(file[1])
                    pred = list(prediction_main[idx])
                    new_img = add_pred2_img(imgs[file[0]], pred, 'main')
            except Exception as e:
                print(e)
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
    Tooth = ['R8', 'R7', 'R6', 'R5', 'R4', 'R3', 'R2', 'R1',
             'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']
    first = True
    ref_hori = []
    for teeth in Tooth:
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

    big_img = np.vstack((big_topimg, ref_hori))
    big_img = np.vstack((big_img, big_botimg))
    big_img = np.hstack((ref_vert, big_img))
    return big_img



