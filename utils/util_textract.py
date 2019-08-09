import cv2
import numpy as np
from sklearn.cluster import KMeans


def table_extract(img, pano_num, output_dir):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 閾値２００以下取り除き、残りのをする
    # 第三引数：閾値の応用法　http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html?highlight=thresh_binary
    _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # ノイズキャンセリング
    # Use shape of 第三引数 to process 第二引数 function
    # MORPH_CLOSE: Closing = Dilation > Erode
    # getStructuringElement(MORPH_ELLIPSE): Create Ellipse array
    # http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html?highlight=morphologyex
    close = cv2.morphologyEx(255 - thr, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # 輪郭探査
    # 第二引数：親輪郭のみ検査　http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contours_hierarchy/py_contours_hierarchy.html?highlight=findcontour
    # 第三引数：輪郭の線資料検査　http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html?highlight=chain_approx_none
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 輪郭に囲まれたエリア大きさを閾値で小さ目のをとりのぞく
    areathr = 20000  # エリア閾値
    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > areathr:
            i = i + 1
            # 奇妙輪郭を取り外す
            approx_cnt = approx_rec(cnt)
            x, y, width, height = cv2.boundingRect(approx_cnt)
            img_crop = img[y:y + height - 1, x:x + width - 1]
            img_scaled = cv2.resize(img_crop, dsize=(2000, 475))
            cv2.imwrite('{}/PNum{:04d}_{}.png'.format(output_dir, pano_num, i),
                        img_scaled)


# 輪郭の近似
# http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
def approx_rec(contour):
    epsilon = 0.2*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    return approx


def check_filenum(file_name):
    # Find Key [DASH] in file name
    dash_idx = [i for i,x in enumerate(file_name) if x is '-']
    start_num = file_name[2:dash_idx[0]]
    inchunk_num = file_name[(dash_idx[1]+1):-4]
    # Can be checked by : print(start_num+' '+inchunk_num)
    return int(start_num) + int(inchunk_num) - 1  # Remove both heads


# Cell Extraction
#############
def outline_parser(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Decide kernal size (Noise Canceling)
    kernel_length = np.array(img).shape[1] // 100

    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    kernel_hori = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_vert = cv2.morphologyEx(255 - thr, cv2.MORPH_OPEN, kernel_vert, iterations=3)
    img_hori = cv2.morphologyEx(255 - thr, cv2.MORPH_OPEN, kernel_hori, iterations=3)

    # Weightvertical and horizontal
    alpha = 0.5
    beta = 1 - alpha

    img_f = cv2.addWeighted(img_vert, alpha,
                            img_hori, beta, 0.0)
    img_f = cv2.erode(~img_f, kernel, iterations=2)
    _, img_f = cv2.threshold(img_f, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return img_f, kernel_length


# Ref:https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/?source=post_page---------------------------
def sort_contours(cnts):
    boundingboxes = [cv2.boundingRect(c) for c in cnts]
    boundingboxes = colpos_adjuster(boundingboxes)
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingboxes), key=lambda x: (x[1][0], x[1][1])))
    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


# Modify x axis to avoid x caused error
def colpos_adjuster(boundingboxes):
    # Using K means to group x axis, 18 column
    kmeans = KMeans(n_clusters=18)
    # Extract x axis only
    x = np.asarray([float(x[0]) for x in boundingboxes], dtype=int)
    # KMeans necessary
    x = x.reshape(-1, 1)
    group_idx = kmeans.fit_predict(x)
    groups = set(group_idx)

    # For every kmean group, reset x axis to mean value within group
    for group in groups:
        x[group_idx == group] = int(np.mean(x[group_idx == group]))

    # chang back to list
    x_mod = x.reshape(1, -1).tolist()

    # Type modify, avoid cant change tuple
    boundingboxes = list(boundingboxes)
    for i, x in enumerate(boundingboxes):
        l_rec = list(boundingboxes[i])
        l_rec[0] = x_mod[0][i]
        boundingboxes[i] = l_rec

    return boundingboxes


def find_contours(img):
    cnt, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cnt


def intable_parser(img, bounding_boxes, pano_num, dir_output, kernel_length):
    nrow = 1
    ncol = 1
    pre_x = 0
    for box in bounding_boxes:
        x, y, w, h = box

        # Remove small cell
        if (w > 60 and h > 25) is False:
            continue
        if w * h > 30000:
            continue

        # Add Col change mechanics
        # order of list is sorted by column and by row (e.g. C1R1, C1R2, C1R3, C2R1, C2R2, ..)
        if x > (pre_x + kernel_length):
            ncol = ncol + 1
            nrow = 1

        # Remove constant cell
        if (ncol in [1, 2]) or (nrow == 5):
            nrow += 1
            pre_x = x
            continue

        # Check code
        ###
        # print('x:{} y:{} w:{} h:{}'.format(str(x), str(y), str(w), str(h)))
        # print('ncol:{} nrow:{} prex:{} klen:{}'.format(str(ncol), str(nrow),
        #                                                                  str(pre_x), str(kernel_length)))
        # print('')
        ###
        new_img = img[y:y + h, x:x + w]
        tooth_annotation = tooth_annote(ncol, nrow)
        new_img_scaled = cv2.resize(new_img, dsize=(90, 45))
        cv2.imwrite('{}/{}_{}.png'.format(dir_output, pano_num, tooth_annotation), new_img_scaled)
        nrow += 1
        pre_x = x


def tooth_annote(ncol, nrow):
    tooth_idx = {3: 'R8', 4: 'R7', 5: 'R6', 6: 'R5', 7: 'R4', 8: 'R3', 9: 'R2', 10: 'R1',
                 11: 'L1', 12: 'L2', 13: 'L3', 14: 'L4', 15: 'L5', 16: 'L6', 17: 'L7', 18: 'L8'}
    tooth_pos = 'U' if nrow in range(1, 5) else 'D'
    tooth_situation = {1: 'Others', 2: 'ApicalLesion', 3: 'Alveolar', 4: 'Caries',
                       6: 'Caries', 7: 'Alveolar', 8: 'ApicalLesion', 9: 'Others'}

    return tooth_idx[ncol][0] + tooth_pos + tooth_idx[ncol][1] + '_' + tooth_situation[nrow]



