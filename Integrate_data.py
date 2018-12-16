# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 22:52:07 2018

@author: 涂彦伦
"""

import h5py
import numpy as np
if __name__ == '__main__':
    f = h5py.File('ecg_traindata.h5','r')
    data = f['data']
    label = f['label']
    data = data[:][:][:][:]
    label = label[:][:][:][:]
    f.close()
    f = h5py.File('ecg_testdata.h5','r')
    test_data = f['data']
    test_label = f['label']
    test_data = test_data[:][:][:][:]
    test_label = test_label[:][:][:][:]
    f.close()
    data  = np.concatenate([data,test_data])
    label = np.concatenate([label,test_label])
    
    
    f = h5py.File('dataset.h5', 'w')
    f.create_dataset('data', data=data)
    f.create_dataset('label', data=label)
    f.close()
    
    