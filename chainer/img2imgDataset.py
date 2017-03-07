#!/usr/bin/env python

import numpy as np
import chainer
'''
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
'''
import six
import os
from glob import glob
import random

from chainer import cuda, optimizers, serializers, Variable
import cv2

def cvt2YUV(img):
    vers = cv2.__version__.split(".")
    if vers[0] == '3':
        img = cv2.cvtColor( img, cv2.COLOR_RGB2YUV )
    else:
        img = cv2.cvtColor( img, cv2.COLOR_BGR2YUV )
    return img

class ImageAndRefDataset(chainer.dataset.DatasetMixin):

    def __init__(self, paths, root1='./input', root2='./ref', dtype=np.float32):
        self._paths = paths
        self._root1 = root1
        self._root2 = root2
        self._dtype = dtype
        self._testpaths = glob('./../main/datasets/manga/val/*.jpg')

    def __len__(self):
        return len(self._testpaths)

    def get_name(self, i):
        return self._paths[i]

    def get_example(self, i, minimize=False, blur=0, s_size=128):
	i = random.randint(0, len(self._testpaths) - 1)
        path1 = self._testpaths[i]
	print(path1)
        image1, _ = load_image_pair(path1)
        image1 = cv2.resize(image1, (512, 512), interpolation=cv2.INTER_AREA)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        #print("load:" + path1, os.path.isfile(path1), image1 is None)
        image1 = np.asarray(image1, self._dtype)
        #print(image1.shape)
        #print(s_size)

        _image1 = image1.copy()
        if minimize:
            if image1.shape[0] < image1.shape[1]:
                #print("img10<img11")
                s0 = s_size
                s1 = int(float(image1.shape[1]) * (float(s_size) / float(image1.shape[0])))
                s1 = s1 - s1 % 16
                _s0 = 4 * s0
                _s1 = int(float(image1.shape[1]) * ( float(_s0) / float(image1.shape[0])))
                _s1 = (_s1+8) - (_s1+8) % 16
            else:
                #print("img10>=img11")
                s1 = s_size
                #print("s0::%d,%d,%d" % (image1.shape[0],s_size,image1.shape[1]))
                s0 = int(float(image1.shape[0]) * (float(s_size) / float(image1.shape[1])))
                #print("s0=")
                #print(s0)
                #print("s0:%d,s1:%d" % (s0,s1))
                s0 = s0 - s0 % 16
                _s1 = 4 * s1
                _s0 = int(float(image1.shape[0]) * ( float(_s1) / float(image1.shape[1])))
                _s0 = (_s0+8) - (_s0+8) % 16

            _image1 = image1.copy()
            _image1 = cv2.resize(_image1, (_s1, _s0),
                                 interpolation=cv2.INTER_AREA)
            #noise = np.random.normal(0,5*np.random.rand(),_image1.shape).astype(self._dtype)

            if blur > 0:
                blured = cv2.blur(_image1, ksize=(blur, blur))
                image1 = _image1 + blured - 255

            image1 = cv2.resize(image1, (s1, s0), interpolation=cv2.INTER_AREA)

        # image is grayscale
        if image1.ndim == 2:
            image1 = image1[:, :, np.newaxis]
        if _image1.ndim == 2:
            _image1 = _image1[:, :, np.newaxis]

        image1 = np.insert(image1, 1, -512, axis=2)
        image1 = np.insert(image1, 2, 128, axis=2)
        image1 = np.insert(image1, 3, 128, axis=2)

        return image1.transpose(2, 0, 1), _image1.transpose(2, 0, 1), i

def load_image_pair(image_path):
    #print("load_image_pair:%s" % image_path)
    input_img = cv2.imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_color = input_img[:, 0:w2]
    img_line = input_img[:, w2:w]
    return img_line, img_color

class Image2ImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, paths, root1='./input', root2='./terget', dtype=np.float32, leak=(0, 0), root_ref = None, train=False):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root1 = root1
        self._root2 = root2
        self._root_ref = root_ref
        self._dtype = dtype
        self._leak = leak
        self._img_dict = {}
        self._train = train

        self._trainpaths = glob('./../main/datasets/manga/train/*.jpg')

    def set_img_dict(self, img_dict):
        self._img_dict = img_dict

    def get_vec(self, name):
        tag_size = 1539
        v = np.zeros(tag_size).astype(np.int32)
        if name in self._img_dict.keys():
            for i in self._img_dict[name][3]:
                v[i] = 1
        return v

    def __len__(self):
        return len(self._trainpaths)

    def get_name(self, i):
        return self._paths[i]

    def get_example(self, i, minimize=False, log=False, bin_r=0):
        if self._train:
            bin_r = 0.9

        image1, image2 = load_image_pair(self._trainpaths[i])
        image1 = cv2.resize(image1, (128, 128), interpolation=cv2.INTER_AREA)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.resize(image2, (128, 128), interpolation=cv2.INTER_AREA)

        #print("image1:%s" % (image1.shape,))
        #print("image2:%s" % (image2.shape,))

        image2 = cvt2YUV( image2 )

        if self._train and np.random.rand() < 0.2:
            ret, image1 = cv2.threshold(
                image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # add flip and noise
        if self._train:
            if np.random.rand() > 0.5:
                image1 = cv2.flip(image1, 1)
                image2 = cv2.flip(image2, 1)
            if np.random.rand() > 0.9:
                image1 = cv2.flip(image1, 0)
                image2 = cv2.flip(image2, 0)

        image1 = np.asarray(image1, self._dtype)
        image2 = np.asarray(image2, self._dtype)

        if self._train:
            noise = np.random.normal(
                0, 5 * np.random.rand(), image1.shape).astype(self._dtype)
            image1 += noise
            noise = np.random.normal(
                0, 5 * np.random.rand(), image2.shape).astype(self._dtype)
            image2 += noise
            noise = np.random.normal(0, 16)
            image1 += noise
            image1[image1 < 0] = 0

        # image is grayscale
        if image1.ndim == 2:
            image1 = image1[:, :, np.newaxis]
        if image2.ndim == 2:
            image2 = image2[:, :, np.newaxis]

        image1 = np.insert(image1, 1, -512, axis=2)
        image1 = np.insert(image1, 2, 128, axis=2)
        image1 = np.insert(image1, 3, 128, axis=2)

        # randomly add terget image px
        if self._leak[1] > 0:
            image0 = image1
            n = np.random.randint(16, self._leak[1])
            if self._train:
                r = np.random.rand()
                if r < 0.4:
                    n = 0
                elif r < 0.7:
                    n = np.random.randint(2, 16)

            x = np.random.randint(1, image1.shape[0] - 1, n)
            y = np.random.randint(1, image1.shape[1] - 1, n)
            for i in range(n):
                for ch in range(3):
                    d = 20
                    v = image2[x[i]][y[i]][ch] + np.random.normal(0, 5)
                    v = np.floor(v / d + 0.5) * d
                    image1[x[i]][y[i]][ch + 1] = v
                if np.random.rand() > 0.5:
                    for ch in range(3):
                        image1[x[i]][y[i] + 1][ch +
                                               1] = image1[x[i]][y[i]][ch + 1]
                        image1[x[i]][y[i] - 1][ch +
                                               1] = image1[x[i]][y[i]][ch + 1]
                if np.random.rand() > 0.5:
                    for ch in range(3):
                        image1[x[i] + 1][y[i]][ch +
                                               1] = image1[x[i]][y[i]][ch + 1]
                        image1[x[i] - 1][y[i]][ch +
                                               1] = image1[x[i]][y[i]][ch + 1]

        image1 = (image1.transpose(2, 0, 1))
        image2 = (image2.transpose(2, 0, 1))
        #image1 = (image1.transpose(2, 0, 1) -128) /128
        #image2 = (image2.transpose(2, 0, 1) -128) /128

        return image1, image2  # ,vec


class Image2ImageDatasetX2(Image2ImageDataset):

    def get_example(self, i, minimize=False, log=False, bin_r=0):
        image1, image2 = load_image_pair(self._trainpaths[i])
        image1 = cv2.resize(image1, (512, 512), interpolation=cv2.INTER_AREA)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.resize(image2, (512, 512), interpolation=cv2.INTER_AREA)

        #path1 = os.path.join(self._root1, self._paths[i])
        #path2 = os.path.join(self._root2, self._paths[i])
        #image1 = ImageDataset._read_image_as_array(path1, self._dtype)
        #image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        #image2 = cv2.imread(path2, cv2.IMREAD_COLOR)
        image2 = cvt2YUV(image2)
        image2 = np.asarray(image2, self._dtype)
        #name1 = os.path.basename(self._paths[i])
        #vec = self.get_vec(name1)

        # add flip and noise
        if self._train:
            if np.random.rand() > 0.5:
                image1 = cv2.flip(image1, 1)
                image2 = cv2.flip(image2, 1)
            if np.random.rand() > 0.8:
                image1 = cv2.flip(image1, 0)
                image2 = cv2.flip(image2, 0)

        if self._train:
            bin_r = 0.3
        if np.random.rand() < bin_r:
            ret, image1 = cv2.threshold(
                image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        _image1 = image1.copy()
        _image2 = image2.copy()
        image1 = cv2.resize(image1, (128, 128), interpolation=cv2.INTER_AREA)
        image2 = cv2.resize(image2, (128, 128), interpolation=cv2.INTER_AREA)

        image1 = np.asarray(image1, self._dtype)
        _image1 = np.asarray(_image1, self._dtype)

        if self._train:
            noise = np.random.normal(0, 5, image1.shape).astype(self._dtype)
            image1 = image1 + noise
            noise = np.random.normal(0, 5, image2.shape).astype(self._dtype)
            image2 = image2 + noise
            noise = np.random.normal(
                0, 4 * np.random.rand(), _image1.shape).astype(self._dtype)
            noise += np.random.normal(0, 24)
            _image1 = _image1 + noise
            _image1[_image1 < 0] = 0
            _image1[_image1 > 255] = 255

        # image is grayscale
        if image1.ndim == 2:
            image1 = image1[:, :, np.newaxis]
        if image2.ndim == 2:
            image2 = image2[:, :, np.newaxis]
        if _image1.ndim == 2:
            _image1 = _image1[:, :, np.newaxis]
        if _image2.ndim == 2:
            _image2 = _image2[:, :, np.newaxis]

        image1 = np.insert(image1, 1, -512, axis=2)
        image1 = np.insert(image1, 2, 128, axis=2)
        image1 = np.insert(image1, 3, 128, axis=2)
        #test::[ 257.96783447 -512.          128.          128.        ]

        # color hint !!!
        # randomly add terget image px
        if self._leak[1] > 0:
            image0 = image1
            n = np.random.randint(self._leak[0], self._leak[1])
            x = np.random.randint(1, image1.shape[0] - 1, n)
            y = np.random.randint(1, image1.shape[1] - 1, n)
            for i in range(n):
                for ch in range(3):
                    d = 20
                    v = image2[x[i]][y[i]][ch] + np.random.normal(0, 5)
                    #v = np.random.normal(128,40)
                    v = np.floor(v / d + 0.5) * d
                    image1[x[i]][y[i]][ch + 1] = v
                    if np.random.rand() > 0.5:
                        image1[x[i]][y[i] + 1][ch + 1] = v
                        image1[x[i]][y[i] - 1][ch + 1] = v
                    if np.random.rand() > 0.5:
                        image1[x[i] + 1][y[i]][ch + 1] = v
                        image1[x[i] - 1][y[i]][ch + 1] = v

        return image1.transpose(2, 0, 1), image2.transpose(2, 0, 1), _image1.transpose(2, 0, 1), _image2.transpose(2, 0, 1)
