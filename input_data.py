# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:31:51 2017

0:airplane
1:automobile
2:bird
3:cat
4:deer
5:dog
6:frog
7:horse
8:ship
9:truck


@author: zwcong2
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding='iso-8859-1')
    fo.close()
    return dict

def one_hot_vec(label):
    vec = np.zeros(10)
    vec[label] = 1
    return vec

def load_data():
    x_all = []
    y_all = []
    for i in range (5):
        d = unpickle("cifar-10-batches-py/data_batch_" + str(i+1))
        x_ = d['data']
        y_ = d['labels']
        x_all.append(x_)
        y_all.append(y_)

    d = unpickle('cifar-10-batches-py/test_batch')
    x_all.append(d['data'])
    y_all.append(d['labels'])

    x = np.concatenate(x_all) / np.float32(255)
    y = np.concatenate(y_all)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))
    
    pixel_mean = np.mean(x[0:50000],axis=0)
    x -= pixel_mean

    y = list(map(one_hot_vec,y))
    y=np.concatenate(y)
    y=y.reshape((np.int(y.shape[0]/10),10))
    X_train = x[0:10000,:,:,:]
    Y_train = y[0:10000]
    X_test = x[50000:55000,:,:,:]
    Y_test = y[50000:55000]
    
    return X_train, Y_train, X_test, Y_test


class DataSet:
    def __init__(self,images,labels):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        pass
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._images = self.images[idx]
            self._labels = self.labels[idx]
        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self.images[start:self._num_examples]
            labels_rest_part = self.labels[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._images = self.images[idx0]
            self._labels = self.labels[idx0]
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            images_new_part =  self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0),np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end],self._labels[start:end]
   
        

if __name__=='__main__':
    X_train, Y_train, X_test, Y_test = load_data()
    im,la = DataSet(X_train,Y_train).next_batch(128)
    f, a = plt.subplots(2, 10, figsize=(12, 3))
    for i in range(10):
        a[0][i].imshow(im[i, :, :, :])
        a[1][i].imshow(im[i + 10, :, :, :])

    # f.show()
    plt.draw()
    
    
    
    
    
    
    