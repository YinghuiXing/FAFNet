from os import listdir
from torch.nn import functional as F
import cv2
import torch
import numpy as np
import os
import random
import scipy.io as scio
import h5py

import mylib as ml

import torch as t
from torch.utils import data
from PIL import Image

'''
DataSet(object) used when patch has already been cropped and mat format  
DataSet_whole(object)   big mats  that haven't been cropped
'''


class DataSet(object):
    def __init__(self, batch_size, root, data_category, name='train'):
        self.allms = []
        self.allpan = []
        self.allref = []

        self.batch_size = batch_size
        self.name = name
        ''' when it is mat patch'''
        if data_category == 'mat':
            self.allpan, self.allms, self.allref = self.all_data_in(root)
        '''when it is h5'''
        if data_category == 'h5':
            self.allpan, self.allms, self.allref = self.read_data(root)
        self.pan_size = self.allpan[0].shape[1]
        self.ms_size = self.allms[0].shape[1]
        self.ref_size = self.allref[0].shape[1]
        self.band = self.allms[0].shape[2]
        self.ratio = int(self.pan_size / self.ms_size)
        # print(self.pan_size)
        # print(self.ms_size)
        # print(self.ref_size)
        # print(self.band)

        self.data_generator = self.generator()

    '''
    # input
    root: the path of data
    name: 'train' | 'test'  | 'val'
    # output
    list of ms/ref/pan  (b,h,w,c)
    '''

    def all_data_in(self, root):
        if self.name == 'train':
            path_data = os.path.join(root, self.name)

        elif self.name == 'test':
            path_data = os.path.join(root, self.name)

        elif self.name == 'val':
            path_data = os.path.join(root, self.name)
        else:
            print("input wrong ,please choose from train | teat |val ")

        ms_path = os.path.join(path_data, 'ms')
        pan_path = os.path.join(path_data, 'pan')
        ref_path = os.path.join(path_data, 'ref')

        ms_list = os.listdir(ms_path)
        pan_list = os.listdir(pan_path)
        ref_list = os.listdir(ref_path)

        ms_list.sort()
        pan_list.sort()
        ref_list.sort()

        # print('ms_list')
        # print(ms_list)
        # print('pan_list')
        # print(pan_list)
        # print('ref_list')
        # print(ref_list)

        for j in range(len(ms_list)):  # j = 0 ~ num-1

            ms_data = scio.loadmat(os.path.join(ms_path, ms_list[j]))
            in_ms = ms_data['ms']
            # normalization
            in_ms = ml.normalized(in_ms)
            self.allms.append(in_ms)

            pan_data = scio.loadmat(os.path.join(pan_path, pan_list[j]))
            in_pan = pan_data['pan']
            in_pan = ml.normalized(in_pan)
            h, w = in_pan.shape
            in_pan = in_pan.reshape([h, w, 1])  # add one channel
            self.allpan.append(in_pan)

            ref_data = scio.loadmat(os.path.join(ref_path, ref_list[j]))
            in_ref = ref_data['reference']
            in_ref = ml.normalized(in_ref)
            self.allref.append(in_ref)

        return self.allms, self.allpan, self.allref

        # read data from the file produced before

    '''
    # input
    data_save_path: the path of data used
    self.name: 'train' | 'test'  | 'val'
    # output
    numpy of ms/ref/pan  (b,h,w,c)
    '''

    def read_data(self, data_save_path):
        all_pan = []
        all_ms = []

        source_ms_path = os.path.join(data_save_path, self.name)
        img_ms_list = os.listdir(source_ms_path)
        img_ms_list.sort(key=lambda x: int(x.split('.')[0]))
        print('img_ms_list', img_ms_list)

        count = 0
        for ii in img_ms_list:
            if self.name == 'train':
                f = h5py.File(os.path.join(data_save_path, 'train', ii.split('.')[0] + '.h5'), 'r')
                pan = np.array(f['pan_train'])  #
                ms = np.array(f['ms_train'])
                # print('read_data train pan.shape', pan.shape)
                # print('read_data train ms.shape', ms.shape)

            elif self.name == 'test':
                f = h5py.File(os.path.join(data_save_path, 'test', ii.split('.')[0] + '.h5'), 'r')
                pan = np.array(f['pan_test'])
                ms = np.array(f['ms_test'])
                # print('read_data test pan.shape', pan.shape)
                # print('read_data test ms.shape', ms.shape)

            else:
                f = h5py.File(os.path.join(data_save_path, 'val', ii.split('.')[0] + '.h5'), 'r')
                pan = np.array(f['pan_valid'])
                ms = np.array(f['ms_valid'])
            f.close()

            if count == 0:
                all_ms = ms
                all_pan = pan
            else:
                all_ms = np.concatenate((all_ms, ms), axis=0)
                all_pan = np.concatenate((all_pan, pan), axis=0)

            print('read_data  all_ms.shape', all_ms.shape)
            print('read_data  all_pan.shape', all_pan.shape)

            count += 1  #

        '''because there is no ref,so need to process'''
        '''en en en ,Some values are fixed'''
        if self.name == 'test':
            ms_size = 64 * 4
        else:
            ms_size = 64
        ratio = 4
        band = 4
        number = len(all_pan)
        pan_end = np.zeros((number, ms_size, ms_size, 1))
        ms_end = np.zeros((number, ms_size // ratio, ms_size // ratio, band))
        ref_end = np.zeros((number, ms_size, ms_size, band))
        for i in range(len(all_pan)):
            # print("in i",i)
            # print(ref_end[i].shape, self.ms[i].shape)
            ref_end[i] = all_ms[i]

            pan_end[i] = cv2.resize(all_pan[i], (ms_size, ms_size)).reshape(ms_size, ms_size,
                                                                            1)  # gai
            ms_end[i] = cv2.resize(cv2.GaussianBlur(all_ms[i], (5, 5), 2),
                                   (ms_size // ratio, ms_size // ratio))  # gai

        # self.pan_size = pan_end[0].shape[1]
        # self.ms_size = ms_end[0].shape[1]
        # self.ref_size = ref_end[0].shape[1]
        return pan_end, ms_end, ref_end

    '''val (4596, 64, 64, 1)(4596, 16, 16, 4)(4596, 64, 64, 4)'''

    def generator(self):
        if isinstance(self.allms, list):
            num_data = len(self.allms)
        if isinstance(self.allms, np.ndarray):
            num_data = self.allms.shape[0]

        random_index = -1
        while True:
            # doubletensor to FloatTensor
            batch_pan = torch.from_numpy(np.zeros((self.batch_size, 1, self.pan_size, self.pan_size))).type(
                torch.FloatTensor)
            batch_ms = torch.from_numpy(np.zeros((self.batch_size, self.band, self.ms_size, self.ms_size))).type(
                torch.FloatTensor)
            batch_ref = torch.from_numpy(np.zeros((self.batch_size, self.band, self.ref_size, self.ref_size))).type(
                torch.FloatTensor)
            # batch size using

            for i in range(self.batch_size):
                # train can be random ,but val and test should not. change
                if self.name != 'test':
                    random_index = np.random.randint(0, num_data)  # a random int from 0 to num_data-1 ,it will repeat
                    print(random_index)

                else:
                    random_index += 1
                    print(random_index)

                batch_pan[i] = torch.from_numpy(self.allpan[random_index]).permute(2, 0, 1)
                batch_ms[i] = torch.from_numpy(self.allms[random_index]).permute(2, 0, 1)
                batch_ref[i] = torch.from_numpy(self.allref[random_index]).permute(2, 0, 1)
            yield batch_pan, batch_ms, batch_ref


# old1   272 total  train 216    val  28     28


# pan_gan data train 710 val 110 test 80 pan 1024 ms 256 stride 128
# train 12002 val 1583 test 1535     pan 256 ms 64 stride 32
'''
this is for big mat that haven't been cropped
stride is for ms
'''


class DataSet_whole(object):
    def __init__(self, batch_size, source_path, data_save_path, bands, category='train', pan_size=256, ms_size=64,
                 stride=64, ratio=4):
        # pan_size=256, ms_size=64, stride=32, ratio=4
        self.pan_size = pan_size
        self.ms_size = ms_size
        self.batch_size = batch_size
        self.bands = bands
        self.category = category
        self.train_data_save_path = os.path.join(source_path, 'train_gf1_64.h5')
        self.test_data_save_path = os.path.join(source_path, 'test_gf1_64.h5')
        if not os.path.exists(self.train_data_save_path):
            self.make_data(source_path, self.train_data_save_path, stride)
        if not os.path.exists(self.test_data_save_path):
            self.make_data(source_path, self.test_data_save_path, stride * 4)

        self.pan, self.ms = self.read_data(data_save_path)
        # print("read data done")
        # print(type(self.pan))
        # print(self.pan.shape)
        # print(type(self.ms))
        # print(self.ms.shape)
        self.pan, self.ms, self.ref = self.ref_pro(ratio)
        self.data_generator = self.generator()

    # read data from the file produced before
    def read_data(self, path):
        f = h5py.File(path, 'r')
        if self.category == 'train':
            pan = np.array(f['pan_train'])  #
            ms = np.array(f['ms_train'])
        elif self.category == 'test':
            pan = np.array(f['pan_test'])
            ms = np.array(f['ms_test'])
            print('read_data ms.shape', ms.shape)
        else:
            pan = np.array(f['pan_valid'])
            ms = np.array(f['ms_valid'])
        return pan, ms

    # from the original whole image to little image patches
    def make_data(self, source_path, data_save_path, stride):
        pan_train = []
        pan_valid = []
        ms_train = []
        ms_valid = []
        # source_ms_path=os.path.join(source_path, 'MS','1.TIF')
        # source_pan_path=os.path.join(source_path, 'PAN','1.TIF')
        #
        # source_ms_path = os.path.join(source_path, 'MS', '2.mat')
        # source_pan_path = os.path.join(source_path, 'PAN', '2.mat')
        source_ms_path = os.path.join(source_path, 'MS')
        source_pan_path = os.path.join(source_path, 'PAN')

        img_ms_list = os.listdir(source_ms_path)
        img_ms_list.sort(key=lambda x: int(x.split('.')[0]))
        print('img_ms_list', img_ms_list)
        img_pan_list = os.listdir(source_pan_path)
        img_pan_list.sort(key=lambda x: int(x.split('.')[0]))
        print('img_pan_list', img_pan_list)

        for ii in img_ms_list:
            img_ms_path = os.path.join(source_ms_path, ii)
            img_pan_path = os.path.join(source_pan_path, ii)
            print('img_ms_path:', img_ms_path)
            print('img_pan_path:', img_pan_path)
            # crop_to_patch
            if ii == '8.mat':
                self.pan_size = self.pan_size * 4
                self.ms_size = self.ms_size * 4
                stride = stride * 4

                ms_test = self.crop_to_patch(img_ms_path, stride, name='ms')
                pan_test = self.crop_to_patch(img_pan_path, stride, name='pan')
                print('The number of ms patch is: ' + str(len(ms_test)))
                print('The number of pan patch is: ' + str(len(pan_test)))
                print('after crop ms size', ms_test[1].shape)
                pan_test = np.array(pan_test)
                ms_test = np.array(ms_test)
            else:

                all_ms = self.crop_to_patch(img_ms_path, stride, name='ms')
                all_pan = self.crop_to_patch(img_pan_path, stride, name='pan')
                print('The number of ms patch is: ' + str(len(all_ms)))
                # all_img.leng: 5040 all_img[0].shape: (64, 64, 4)
                print('The number of pan patch is: ' + str(len(all_pan)))
                # all_img.leng: 5040 all_img[0].shape: (256, 256)
                # split_data
                pan_train_, pan_valid_, ms_train_, ms_valid_ = self.split_data(all_pan, all_ms)  # pan_train : a list
                print('The number of pan_train patch is: ' + str(len(pan_train)))
                print('The number of ms_train patch is: ' + str(len(ms_train)))
                print('The number of pan_valid patch is: ' + str(len(pan_valid)))
                print('The number of ms_valid patch is: ' + str(len(ms_valid)))
                pan_train.append(pan_train_)
                pan_valid.append(pan_valid_)
                ms_train.append(ms_train_)
                ms_valid.append(ms_valid_)
                # # # change the data type
                # # pan_train = np.array(pan_train)
                # # pan_valid = np.array(pan_valid)
                # # ms_train = np.array(ms_train)
                # # ms_valid = np.array(ms_valid)
                # print("start to write in file")
                # if not os.path.exists(file_name):
                #     f = h5py.File(data_save_path, 'w')
                #
                # else:
                #     f = h5py.File(data_save_path, 'a')

                # f.create_dataset('pan_train', data=pan_train)  # so in the dataset ,pan_train still be a list
                # f.create_dataset('pan_valid', data=pan_valid)
                # f.create_dataset('pan_test', data=pan_test)
                # f.create_dataset('ms_train', data=ms_train)
                # f.create_dataset('ms_valid', data=ms_valid)
                # f.create_dataset('ms_test', data=ms_test)
                # f.close()

        print("start to write in file")
        f = h5py.File(data_save_path, 'w')
        f.create_dataset('pan_train', data=pan_train)
        f.create_dataset('pan_valid', data=pan_valid)
        f.create_dataset('pan_test', data=pan_test)
        f.create_dataset('ms_train', data=ms_train)
        f.create_dataset('ms_valid', data=ms_valid)
        f.create_dataset('ms_test', data=ms_test)
        f.close()

        print("make_data done")

    # crop the big image to little image patches
    def crop_to_patch(self, img_path, stride, name):
        # img=(cv2.imread(img_path,-1)-127.5)/127.5
        # img = self.read_img2(img_path) # pangan data
        img = self.read_img3(img_path, name)
        h = img.shape[0]
        w = img.shape[1]
        print(h)
        print(w)
        all_img = []
        if name == 'ms':
            for i in range(0, h - self.ms_size,
                           stride):  # pan_size=128 ms_size=32 ratio=4 stride=16 之所以不是按照size间隔，是因为切分数据时有重叠
                for j in range(0, w - self.ms_size, stride):
                    img_patch = img[i:i + self.ms_size, j:j + self.ms_size, :]
                    all_img.append(img_patch)
                    if i + self.ms_size >= h:
                        img_patch = img[h - self.ms_size:, j:j + self.ms_size, :]
                        all_img.append(img_patch)
                img_patch = img[i:i + self.ms_size, w - self.ms_size:, :]
                all_img.append(img_patch)
        else:
            for i in range(0, h - self.pan_size, stride * 4):  #
                for j in range(0, w - self.pan_size, stride * 4):  #
                    img_patch = img[i:i + self.pan_size,
                                j:j + self.pan_size]  # .reshape(self.pan_size, self.pan_size,1)

                    all_img.append(img_patch)
                    if i + self.pan_size >= h:
                        img_patch = img[h - self.pan_size:,
                                    j:j + self.pan_size]  # .reshape(self.pan_size,self.pan_size, 1)

                        all_img.append(img_patch)
                img_patch = img[i:i + self.pan_size, w - self.pan_size:]  # .reshape(self.pan_size, self.pan_size, 1)
                all_img.append(img_patch)
        print('all_img.leng:', len(all_img), 'all_img[0].shape:', all_img[0].shape)
        return all_img

    # split all the patches to test/val/train dataset
    # use list to save data, return list
    def split_data(self, all_pan, all_ms):
        ''' all_pan和all_ms均为list'''
        pan_train = []
        pan_valid = []
        # pan_test = []
        # ms_test = []
        ms_train = []
        ms_valid = []
        for i in range(len(all_pan)):
            rand = np.random.randint(0, 100)  # gai
            if rand <= 10:
                pan_valid.append(all_pan[i])
                ms_valid.append(all_ms[i])
            # elif 10 < rand <= 20:
            #     ms_test.append(all_ms[i])
            #     pan_test.append(all_pan[i])
            else:
                ms_train.append(all_ms[i])
                pan_train.append(all_pan[i])
        print('pan_train.leng:', len(pan_train), 'pan_train[0].shape:', pan_train[0].shape)
        print('pan_valid.leng:', len(pan_valid), 'pan_valid[0].shape:', pan_valid[0].shape)
        print('ms_train.leng:', len(ms_train), 'ms_train[0].shape:', ms_train[0].shape)
        print('ms_valid.leng:', len(ms_valid), 'ms_valid[0].shape:', ms_valid[0].shape)
        '''
        pan_train.leng: 4505 pan_train[0].shape: (256, 256)
        pan_valid.leng: 535 pan_valid[0].shape: (256, 256)
        ms_train.leng: 4505 ms_train[0].shape: (64, 64, 4)
        ms_valid.leng: 535 ms_valid[0].shape: (64, 64, 4)
        '''
        return pan_train, pan_valid, ms_train, ms_valid  # , pan_test, ms_test

    def read_img(self, path, name):
        data = gdal.Open(path)
        w = data.RasterXSize
        h = data.RasterYSize
        img = data.ReadAsArray(0, 0, w, h)
        if name == 'ms':
            img = np.transpose(img, (1, 2, 0))
        img = (img - 1023.5) / 1023.5
        return img

    def read_img2(self, path):
        img = scio.loadmat(path)['I']
        img = ml.normalized(img)  # zy
        # img = (img - 127.5) / 127.5 # from pangan
        # 你说它减去127.5再除以127.5有什么好处呢，127.5第一个像是均值，减均值除以最大值，归一化
        return img

    def read_img3(self, path, name):
        if name == 'ms':
            # img = scio.loadmat(path)['ms']
            f = h5py.File(path, 'r')  # return 'File' object
            img = np.transpose(np.array(f['ms']))  #
            print('原始的MAT读出来的格式:', type(img), 'img.shape:', img.shape)
            # 原始的MAT读出来的格式: <class 'numpy.ndarray'> img.shape: (4500, 4548, 4)
            img = ml.normalized(img)  # normalization
            # img = (img - 1023.5) / 1023.5
        else:
            # img = scio.loadmat(path)['pan']
            f = h5py.File(path, 'r')  # return 'File' object
            img = np.transpose(np.array(f['pan']))  #
            img = ml.normalized(img)  # normalization
            # img = (img - 1023.5) / 1023.5
        # img = (img - 127.5) / 127.5 # from pangan
        # 你说它减去127.5再除以127.5有什么好处呢，127.5第一个像是均值，减均值除以最大值，归一化

        return img

    '''in img list   out list (length,h,w,c)'''

    def ref_pro(self, ratio):
        ref_end = self.ms
        pan_end = np.zeros((self.pan.shape[0], self.ms_size, self.ms_size, 1))
        ms_end = np.zeros((self.pan.shape[0], self.ms_size // ratio, self.ms_size // ratio, self.bands))

        for i in range(self.pan.shape[0]):
            # print("in")
            pan_end[i] = cv2.resize(self.pan[i], (self.ms_size, self.ms_size)).reshape(self.ms_size, self.ms_size,
                                                                                       1)  # gai
            ms_end[i] = cv2.resize(cv2.GaussianBlur(self.ms[i], (5, 5), 2),
                                   (self.ms_size // ratio, self.ms_size // ratio))  # gai

        # print("done ,now we have pan ms ref")
        # print(pan_end.shape)
        # print(ms_end.shape)
        self.pan_size = pan_end[0].shape[1]
        self.ms_size = ms_end[0].shape[1]
        self.ref_size = ref_end[0].shape[1]
        return pan_end, ms_end, ref_end

    def generator(self):
        # num_data = len(self.ms)
        num_data = self.pan.shape[0]  # from pangan
        random_index = -1
        while True:
            # doubletensor to FloatTensor

            batch_pan = torch.from_numpy(np.zeros((self.batch_size, 1, self.pan_size, self.pan_size))).type(
                torch.FloatTensor)
            batch_ms = torch.from_numpy(np.zeros((self.batch_size, self.bands, self.ms_size, self.ms_size))).type(
                torch.FloatTensor)
            batch_ref = torch.from_numpy(np.zeros((self.batch_size, self.bands, self.ref_size, self.ref_size))).type(
                torch.FloatTensor)

            # batch size using
            for i in range(self.batch_size):
                # train can be random ,but val and test should not. change

                if self.category != 'test':
                    random_index = np.random.randint(0, num_data)  # a random int from 0 to num_data-1 ,it will repeat
                    print(random_index)

                else:
                    random_index += 1
                    print(random_index)
                batch_pan[i] = torch.from_numpy(self.pan[random_index]).permute(2, 0, 1)
                batch_ms[i] = torch.from_numpy(self.ms[random_index]).permute(2, 0, 1)
                batch_ref[i] = torch.from_numpy(self.ref[random_index]).permute(2, 0, 1)
            yield batch_pan, batch_ms, batch_ref


'''
# from the original whole image to little image patches
    def make_data(self, source_path, data_save_path, stride):
        # source_ms_path=os.path.join(source_path, 'MS','1.TIF')
        # source_pan_path=os.path.join(source_path, 'PAN','1.TIF')
        #
        # source_ms_path = os.path.join(source_path, 'MS', '2.mat')
        # source_pan_path = os.path.join(source_path, 'PAN', '2.mat')       

        # crop_to_patch
        all_pan = self.crop_to_patch(source_pan_path, stride, name='pan')
        all_ms = self.crop_to_patch(source_ms_path, stride, name='ms')
        print('The number of ms patch is: ' + str(len(all_ms)))
        print('The number of pan patch is: ' + str(len(all_pan)))
        # split_data
        pan_train, pan_valid, ms_train, ms_valid , pan_test, ms_test= self.split_data(all_pan, all_ms)
        print('The number of pan_train patch is: ' + str(len(pan_train)))
        print('The number of pan_valid patch is: ' + str(len(pan_valid)))
        print('The number of pan_test patch is: ' + str(len(pan_test)))
        print('The number of ms_train patch is: ' + str(len(ms_train)))
        print('The number of ms_valid patch is: ' + str(len(ms_valid)))
        print('The number of ms_test patch is: ' + str(len(ms_test)))

        # change the data type
        pan_train = np.array(pan_train)
        pan_valid = np.array(pan_valid)
        pan_test = np.array(pan_test)
        ms_train = np.array(ms_train)
        ms_valid = np.array(ms_valid)
        ms_test = np.array(ms_test)

        print("start to write in file")
        f = h5py.File(data_save_path, 'w')
        f.create_dataset('pan_train', data=pan_train)
        f.create_dataset('pan_valid', data=pan_valid)
        f.create_dataset('pan_test', data=pan_test)
        f.create_dataset('ms_train', data=ms_train)
        f.create_dataset('ms_valid', data=ms_valid)
        f.create_dataset('ms_test', data=ms_test)
        print("make_data done")
'''


class Datain(data.Dataset):
    def __init__(self, root, ratio):  # root: data in path
        self.ms_save_path = os.path.join(root, 'ms')
        self.pan_save_path = os.path.join(root, 'pan')

        imgs_ms = os.listdir(self.ms_save_path)  # the name list of the images
        imgs_ms.sort(key=lambda x: int(x.split('.')[0]))
        self.imgs_ms = [os.path.join(root, 'ms', img) for img in imgs_ms]  # the whole path of images list
        imgs_pan = os.listdir(self.pan_save_path)  # the name list of the images
        imgs_pan.sort(key=lambda x: int(x.split('.')[0]))
        self.imgs_pan = [os.path.join(root, 'pan', img) for img in imgs_pan]  # the whole path of images list
        # print('imgs_ms : ', imgs_ms)
        # print('self.imgs_ms : ', self.imgs_ms)
        # print('imgs_pan : ', imgs_pan)
        # print('self.imgs_pan : ', self.imgs_pan)
        self.ratio = ratio

    def __getitem__(self, item):
        img_path_ms = self.imgs_ms[item]  # the path of the item-th img
        # print('img_path_ms: ', img_path_ms)

        # with open('train_WV2.txt', 'a') as f:
        #     f.write('img_path_ms: ' + img_path_ms + '\n')
        # f.close()

        img_ms = scio.loadmat(img_path_ms)['ms']  # <class 'numpy.ndarray'>
        img_path_pan = self.imgs_pan[item]
        img_pan = scio.loadmat(img_path_pan)['pan']
        # print(img_ms.shape)
        # print(img_pan.shape)
        # print(type(img_pan))

        # img_ms = ml.normalized(img_ms)  # normalization
        # img_pan = ml.normalized(img_pan)  # normalization

        img_ms = img_ms / 2047.   # normalization
        img_pan = img_pan / 2047. # normalization

        # img_ms = (img_ms - 1023.5) / 1023.5  # normalization
        # img_pan = (img_pan - 1023.5) / 1023.5 # normalization

        '''resize ,from pan and ms ,produce ref, pan, ms; numpy format'''
        ms_size = img_ms.shape[1]
        # print(ms_size)
        ref_np = img_ms
        # gai
        '''To shrink an image, it will generally look best with #INTER_AREA interpolation'''
        pan_np = cv2.resize(img_pan, (ms_size, ms_size), interpolation=cv2.INTER_AREA).reshape(ms_size, ms_size, 1)
        ms_np = cv2.resize(cv2.GaussianBlur(img_ms, (5, 5), 2),
                           (ms_size // self.ratio, ms_size // self.ratio), interpolation=cv2.INTER_AREA)  # gai
        '''
        shape
        (256, 256, 4)
        (256, 256, 1)
        (64, 64, 4)      
        '''

        '''data format change'''
        ref_torch_hwc = torch.from_numpy(ref_np).type(torch.FloatTensor)
        pan_torch_hwc = torch.from_numpy(pan_np).type(torch.FloatTensor)
        ms_torch_hwc = torch.from_numpy(ms_np).type(torch.FloatTensor)

        '''channels change position'''
        ref = ref_torch_hwc.permute(2, 0, 1)
        pan = pan_torch_hwc.permute(2, 0, 1)
        ms = ms_torch_hwc.permute(2, 0, 1)

        '''
        shape
        torch.Size([4, 256, 256])
        torch.Size([1, 256, 256])
        torch.Size([4, 64, 64])
        '''
        return ref, pan, ms  # 返回图片对应的tensor及其标签

    def __len__(self):
        return len(self.imgs_ms)

class Datain_testfull(data.Dataset):
    def __init__(self, root, ratio):  # root: data in path
        self.ms_save_path = os.path.join(root, 'ms')
        self.pan_save_path = os.path.join(root, 'pan')

        imgs_ms = os.listdir(self.ms_save_path)  # the name list of the images
        imgs_ms.sort(key=lambda x: int(x.split('.')[0]))
        self.imgs_ms = [os.path.join(root, 'ms', img) for img in imgs_ms]  # the whole path of images list
        imgs_pan = os.listdir(self.pan_save_path)  # the name list of the images
        imgs_pan.sort(key=lambda x: int(x.split('.')[0]))
        self.imgs_pan = [os.path.join(root, 'pan', img) for img in imgs_pan]  # the whole path of images list
        # print('imgs_ms : ', imgs_ms)
        # print('self.imgs_ms : ', self.imgs_ms)
        # print('imgs_pan : ', imgs_pan)
        # print('self.imgs_pan : ', self.imgs_pan)
        self.ratio = ratio

    def __getitem__(self, item):
        img_path_ms = self.imgs_ms[item]  # the path of the item-th img
        print('img_path_ms: ', img_path_ms)
        img_ms = scio.loadmat(img_path_ms)['ms']  # <class 'numpy.ndarray'>
        img_path_pan = self.imgs_pan[item]
        img_pan = scio.loadmat(img_path_pan)['pan']
        # print(img_ms.shape)
        # print(img_pan.shape)
        # print(type(img_pan))

        # img_ms = ml.normalized(img_ms)  # normalization
        # img_pan = ml.normalized(img_pan)  # normalization

        img_ms = img_ms / 2047.  # normalization
        img_pan = img_pan / 2047.  # normalization


        '''data format change'''
        pan_torch_hwc = torch.from_numpy(img_pan).type(torch.FloatTensor).unsqueeze(0)
        ms_torch_hwc = torch.from_numpy(img_ms).type(torch.FloatTensor)

        '''channels change position'''
        pan = pan_torch_hwc #.permute(1, 2, 0)
        ms = ms_torch_hwc.permute(2, 0, 1)



        # pan = pan[:, 200:800, 200:800]  # .permute(1, 2, 0)
        # _, ms_h, ms_w = ms.shape
        #
        # ms = ms[:, 50:200, 50:200]

        '''
        shape
        torch.Size([1, 1024, 1024])
        torch.Size([4, 256, 256])
        '''
        return pan, ms  # 返回图片对应的tensor及其标签

    def __len__(self):
        return len(self.imgs_ms)

if __name__ == '__main__':
    # dataset = DataSet(1, 'data', 'val')
    # DataGenerator = dataset.data_generator
    # for i in range(5):
    #     pan_batch, ms_batch, ref_batch = next(DataGenerator)
    #     print(pan_batch.shape)
    #     print(ms_batch.shape)
    #     print(ref_batch.shape)
    # print('_'*40)
    # dataset_train = DataSet(4, 'data', 'train')
    # DataGenerator = dataset_train.data_generator
    # for j in range(5):
    #     pan_batch, ms_batch, ref_batch = next(DataGenerator)
    #     print(pan_batch.shape)
    #     print(ms_batch.shape)
    #     print(ref_batch.shape)

    # dataset = DataSet_whole(2, 'data/pan_gan_data', 'data/train_qk.h5', 4 ,'test')#, 7600, 1900, 950
    # DataGenerator = dataset.data_generator
    # for i in range(5):
    #     pan_batch, ms_batch, ref_batch= next(DataGenerator)
    #     print(pan_batch.shape)
    #     print(ms_batch.shape)
    #     print(ref_batch.shape)
    #     print('_'*40)

    # done ,now we have pan ms ref
    # (1535, 64, 64, 1)
    # (1535, 16, 16, 4)
    #     dataset = DataSet_whole(1, 'data/pan_gan_data', 'data/train_qk.h5', 4, 'test')  # , 7600, 1900, 950
    #     pan_end=dataset.pan
    #     ms_end=dataset.ms
    #     ref_end = dataset.ref
    #
    #     print(pan_end.shape)
    #     print(ms_end.shape)
    #     print(ref_end.shape)
    #     print(type(pan_end))

    # dataset = DataSet(1, 'data/GF1_patch', 'h5',  'train')  # , 7600, 1900, 950(self, batch_size, root, data_category, name='train'):
    # DataGenerator = dataset.data_generator
    # for i in range(3):
    #     pan_batch, ms_batch, ref_batch = next(DataGenerator)
    #     print(pan_batch.shape)
    #     print(ms_batch.shape)
    #     print(ref_batch.shape)
    #     # print('_'*40)

    dogcat = Datain_testfull('data/GF1_mat/test', 4)
    pan, ms = dogcat[0]

    print(pan.shape)
    print(ms.shape)
    print(len(dogcat))

'''
imgs :  ['309.mat', '403.mat', '176.mat', '247.mat', '191.mat', '509.mat', '595.mat', ......]
self.imgs :  ['data/GF1_mat/test/309.mat', 'data/GF1_mat/test/403.mat', 'data/GF1_mat/test/176.mat',......]
612

final img shape:
(256, 256, 4)
(256, 256, 1)
(64, 64, 4)
'''
