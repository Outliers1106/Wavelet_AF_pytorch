# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 23:14:25 2018

@author: 涂彦伦
"""

'''
to make the number samples to be similar, expand the dataset
'''
import h5py
import numpy as np
import random

def split_class(data,label):
    dataN = np.zeros([1,1,12,18286])
    dataA = np.zeros([1,1,12,18286])
    dataO = np.zeros([1,1,12,18286])
    dataNoisy = np.zeros([1,1,12,18286])
    labelN = np.zeros([1,4,1,1])
    labelA = np.zeros([1,4,1,1])
    labelO = np.zeros([1,4,1,1])
    labelNoisy = np.zeros([1,4,1,1])
    for i in range(label.shape[0]):
        #get index of label[i] (4x1) where number = 1 
        index = np.where(label[i]==1)[0][0]
        if index == 0:
            dataN = np.concatenate([dataN,np.expand_dims(data[i],axis=0)])
            labelN = np.concatenate([labelN,np.expand_dims(label[i],axis=0)])
        elif index == 1:
            dataA = np.concatenate([dataA,np.expand_dims(data[i],axis=0)])
            labelA = np.concatenate([labelA,np.expand_dims(label[i],axis=0)])
        elif index == 2:
            dataO = np.concatenate([dataO,np.expand_dims(data[i],axis=0)])
            labelO = np.concatenate([labelO,np.expand_dims(label[i],axis=0)])
        else:
            dataNoisy = np.concatenate([dataNoisy,np.expand_dims(data[i],axis=0)])
            labelNoisy = np.concatenate([labelNoisy,np.expand_dims(label[i],axis=0)])
    dataN = np.delete(dataN,0,axis=0)
    dataA = np.delete(dataA,0,axis=0)
    dataO = np.delete(dataO,0,axis=0)
    dataNoisy = np.delete(dataNoisy,0,axis=0)
    labelN = np.delete(labelN,0,axis=0)
    labelA = np.delete(labelA,0,axis=0)
    labelO = np.delete(labelO,0,axis=0)
    labelNoisy = np.delete(labelNoisy,0,axis=0)
    return dataN,dataA,dataO,dataNoisy,labelN,labelA,labelO,labelNoisy

def data_expand(dataN,dataA,dataO,dataNoisy,labelN,labelA,labelO,labelNoisy):
    #A -> 7A
    dataA = np.row_stack([dataA,dataA,dataA,dataA,dataA,dataA,dataA])
    labelA = np.row_stack([labelA,labelA,labelA,labelA,labelA,labelA,labelA])
    #O -> 2O
    dataO = np.row_stack([dataO,dataO])
    labelO = np.row_stack([labelO,labelO])
    #Noisy ->18Noisy
    dataNoisy = np.row_stack([dataNoisy,dataNoisy,dataNoisy])
    labelNoisy = np.row_stack([labelNoisy,labelNoisy,labelNoisy])
    dataNoisy = np.row_stack([dataNoisy,dataNoisy,dataNoisy])
    labelNoisy = np.row_stack([labelNoisy,labelNoisy,labelNoisy])
    dataNoisy = np.row_stack([dataNoisy,dataNoisy])
    labelNoisy = np.row_stack([labelNoisy,labelNoisy])
    return dataN,dataA,dataO,dataNoisy,labelN,labelA,labelO,labelNoisy

if __name__=='__main__':
    f = h5py.File('dataset.h5','r')
    data = f['data']
    label = f['label']
    data = data[:][:][:][:]
    label = label[:][:][:][:]
    f.close()
    dataN,dataA,dataO,dataNoisy,labelN,labelA,labelO,labelNoisy=split_class(data,label)
    dataN,dataA,dataO,dataNoisy,labelN,labelA,labelO,labelNoisy=data_expand(dataN,dataA,dataO,dataNoisy,labelN,labelA,labelO,labelNoisy)
    data = np.row_stack(dataN,dataA,dataO,dataNoisy])
    label = np.row_stack([labelN,labelA,labelO,labelNoisy])
        