import numpy as np
import cv2
import os
import re
import random
from math import ceil


class DataHouse:
    def __init__(self):
        self.mnist = None
        self.local_marked_path = None
        self.external_data_path = None
        self.augment_data_path = None
        self.pred_data_path = None

        self.datam_x = None
        self.datam_y = None
        self.datas_x = None
        self.datas_y = None

        self.datam_pred = None
        self.listm_pred = []
        self.datas_pred = None
        self.lists_pred = []

        self.first_train = True

        self.dish_order = []
        self.dish_num = 0
        self.iter_num = 0

    def reset(self):
        self.first_train = True
        self.dish_order = []
        self.dish_num = 0
        self.iter_num = 0

    def set_iter(self, num):
        self.iter_num = num

    @staticmethod
    def _npclass(length, num):
        list_class = [0.]*length
        list_class[num] += 1
        return np.array(list_class)

    @staticmethod
    def _imgtune(img, resize28):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1]
        img = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        img_f = img.astype('float32') / 255
        if resize28:
            img_f = cv2.resize(img_f, (28, 28))
        return img_f

    @staticmethod
    def _augimgprep(dir_path, count):
        augimg_lists = os.listdir(dir_path)
        random.shuffle(augimg_lists)
        imgs = [cv2.imread('{}{}'.format(dir_path, x)) for x in augimg_lists[:count]]
        imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs]
        return imgs

    @staticmethod
    def _mainsub_extraction(ms, files_list):
        statuses = None
        if ms is 'main':
            statuses = ["Alveolar", "Caries"]
        elif ms is 'sub':
            statuses = ["ApicalLesion", "Others"]
        files_sub = [file for status in statuses for file in files_list if status in file]
        return files_sub

    @staticmethod
    def _concatstack(arrays):
        concat_array = arrays[0]
        for array in arrays[1:]:
            concat_array = np.vstack((concat_array, array))
        return concat_array

    @staticmethod
    def _data_check(data, ms):
        thres = 784 if ms is 'main' else 4050
        if data.shape[1] == thres & all([x > 0.4 for x in list(data.mean(axis=1))]):
            return True
        else:
            return False

    def _augunitsprep(self, count, process_list):
        aug_imgs = []
        aug_labels = []
        for tupl in process_list:
            dir_name, class_len, class_stat = tupl
            aug_imgs += self._augimgprep('{}/{}/output/'.format(self.augment_data_path, dir_name),
                                         count)
            aug_labels += [self._npclass(class_len, class_stat)] * count
        return aug_imgs, aug_labels

    def _mnistdata_prep(self):
        mnist_imgs_all, mnist_labels_all = self.mnist.train.next_batch(50000)
        ng_labels = [self._npclass(10, 0), self._npclass(10, 8), self._npclass(10, 9)]
        ng_idx = []

        for i, label in enumerate(mnist_labels_all):
            euc_dist = [np.linalg.norm(x) for x in (ng_labels - label)]
            if all([x != 0 for x in euc_dist]):
                ng_idx.append(i)

        mnist_labels_trim = mnist_labels_all[ng_idx]
        mnist_train_labels = np.array([np.delete(a, [8, 9]) for a in mnist_labels_trim])
        mnist_train_imgs = mnist_imgs_all[ng_idx]
        return mnist_train_imgs, mnist_train_labels

    def _localdata_prep(self, labels, ms):
        size = 784 if ms is 'main' else 4050
        resize28 = True if ms is 'main' else False
        local_imgs = []
        local_labels = []

        files_all = os.listdir(self.local_marked_path)
        labels_original = np.zeros(len(labels), dtype=np.float32)
        files_sub = [file for label in labels for file in files_all if label in file[0]]

        for file in files_sub:
            img = cv2.imread(self.local_marked_path + '/' + file)
            img = self._imgtune(img, resize28)
            local_imgs.append(img)

            label_str = re.split('\.| ', file)[0]
            label_idx = labels.index(label_str)
            label = np.copy(labels_original)
            label[label_idx] += 1
            local_labels.append(label)

        local_imgs = np.array(local_imgs).reshape([-1, size])
        local_labels = np.array(local_labels)
        return local_imgs, local_labels

    def _externaldata_prep(self, count):
        p_dir = self.external_data_path
        c_dir = [x for x in os.listdir(p_dir) if re.match('[0-9]{4}', x)]
        c_dir = c_dir + ['UNCATEGORIZED']
        numdir = ['2', '3', '4', '5', '6']
        label_order = ['r', '1', '2', '3', '4', '5', '6', '7']
        label_raw = [0., 0., 0., 0., 0., 0., 0., 0.]
        external_imgs = []
        external_labels = []

        for sample in c_dir:
            nums = numdir + ['1', '7'] if 'US' in sample else numdir
            path = p_dir + sample + '/'
            for num in nums:
                files = os.listdir(path + '/' + num)
                random.shuffle(files)
                label = label_raw.copy()
                label[label_order.index(num)] += 1
                for file in files[:count]:
                    img = cv2.imread(path + '/' + num + '/' + file)
                    img = self._imgtune(img, True)
                    external_imgs.append(img)
                    external_labels.append(label)

        external_imgs = np.array(external_imgs).reshape([-1, 784])
        external_labels = np.array(external_labels)
        return external_imgs, external_labels

    def _augmentdatam_prep(self, count):
        # tupleformat -> (foldername, classlength, class)
        aug_main_stat = [('blank28', 8, 0), ('1', 8, 1), ('7', 8, 7)]
        aug_imgs, aug_labels = self._augunitsprep(count, aug_main_stat)

        aug_imgs = np.array(aug_imgs).reshape([-1, 784]) / 255
        aug_labels = np.array(aug_labels)
        return aug_imgs, aug_labels

    def _augmentdatas_prep(self, count):
        # class_sub -> ['blank', 'slash', 'exist', 'implant', 'etc']
        # tupleformat -> (foldername, classlength, class)
        aug_main_stat = [('blank45', 5, 0), ('slice', 5, 1), ('exist', 5, 2),
                         ('implant', 5, 3), ('etc', 5, 4)]
        aug_imgs, aug_labels = self._augunitsprep(count, aug_main_stat)

        aug_imgs = np.array(aug_imgs).reshape([-1, 4050]) / 255
        aug_labels = np.array(aug_labels)
        return aug_imgs, aug_labels

    def _preddata_load(self, path, ms, files_list):
        size = 784 if ms is 'main' else 4050
        resize28 = True if ms is 'main' else False
        files_sub = self._mainsub_extraction(ms, files_list)
        imgs = [cv2.imread('{}{}'.format(path, file)) for file in files_sub]
        imgs = [self._imgtune(img, resize28) for img in imgs]
        imgs = np.array(imgs).reshape([-1, size])
        return imgs, files_sub

    def _preddatams_prep(self, ms):
        path = self.pred_data_path
        first_pred = True
        pred_data = None
        pred_list = None
        print('Loading Prediction Data for {}...'.format(ms))
        list_pano = [x for x in os.listdir(path) if "PNum" in x]
        for directory in list_pano:
            list_img = os.listdir(path + directory)
            if first_pred:
                pred_data, pred_list = self._preddata_load(path + directory + '/', ms, list_img)
                first_pred = False
                continue
            new_data, new_list = self._preddata_load(path + directory + '/', ms, list_img)
            pred_data = np.vstack((pred_data, new_data))
            pred_list += new_list
        return pred_data, pred_list

    def train_data_prep(self, aug_count=0, ext_count=0):
        print('Loading MNIST...')
        data_x_mnist, data_y_mnist = self._mnistdata_prep()
        print('Loading Local Data...')
        data_x_local, data_y_local = self._localdata_prep(['r', '1', '2', '3', '4', '5', '6', '7'], 'main')
        print('Loading Augment Main Data...')
        data_x_aug, data_y_aug = self._augmentdatam_prep(aug_count)
        print('Loading External Data...')
        data_x_external, data_y_external = self._externaldata_prep(ext_count)

        self.datam_x = self._concatstack((data_x_local, 1 - data_x_mnist, data_x_external, data_x_aug))
        self.datam_y = self._concatstack((data_y_local, data_y_mnist, data_y_external, data_y_aug))

        print('Loading Local KANJI Data...')
        datas_x_local, datas_y_local = self._localdata_prep(['r', 'd', 'h', 'i', 'e'], 'sub')
        print('Loading Augment KANJI Data...')
        datas_x_aug, datas_y_aug = self._augmentdatas_prep(aug_count)
        self.datas_x = np.vstack((datas_x_local, datas_x_aug))
        self.datas_y = np.vstack((datas_y_local, datas_y_aug))

    def pred_data_prep(self):
        self.datam_pred, self.listm_pred = self._preddatams_prep('main')
        self.datas_pred, self.lists_pred = self._preddatams_prep('sub')

    def alldata_check(self):
        name_data = ['datam_x', 'datas_x', 'datam_pred', 'datas_pred']
        test_check = [self._data_check(self.datam_x, 'main'), self._data_check(self.datas_x, 'sub'),
                      self._data_check(self.datam_pred, 'main'), self._data_check(self.datas_pred, 'sub')]
        if all(test_check):
            print('Shape and value range of all data checked, readu to ride!')
        else:
            print('{} has different shape or value, checked again.'.format(name_data[test_check.index(False)]))

    def serve_dish_train(self, mode, batch_size):
        data_size = None
        batch_x, batch_y = None, None

        if self.first_train:
            total_length = batch_size * self.iter_num
            if mode is 'main': data_size = self.datam_x.shape[0]
            elif mode is 'sub': data_size = self.datas_x.shape[0]

            epoch_count = ceil(total_length / data_size)
            for i in range(epoch_count):
                inepoch_order = list(range(data_size))
                random.shuffle(inepoch_order)
                self.dish_order += inepoch_order
            self.first_train = False

        rand_batch_list = self.dish_order[self.dish_num: (self.dish_num + batch_size)]
        self.dish_num += batch_size

        if mode is 'main':
            batch_x = [self.datam_x[i] for i in rand_batch_list]
            batch_y = [self.datam_y[i] for i in rand_batch_list]
        elif mode is 'sub':
            batch_x = [self.datas_x[i] for i in rand_batch_list]
            batch_y = [self.datas_y[i] for i in rand_batch_list]

        return batch_x, batch_y
