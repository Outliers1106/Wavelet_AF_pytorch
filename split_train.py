import h5py
import numpy as np
# split trainset to 4 classes
# N: 4060 A 606 O 1932 Noisy 223
f = h5py.File('./hdf5file/ecg_traindata.h5', 'r')
data = f['data']
label = f['label']
label_iter = label[:]
label_iter = label_iter.squeeze(3)
label_iter = label_iter.squeeze(2)
N,A,O,Noisy=[0,0,0,0]
for i in range(label_iter.shape[0]):
    if label_iter[i][0] == 1:
        N = N+1
    elif label_iter[i][1] == 1:
        A = A+1
    elif label_iter[i][2] == 1:
        O = O+1
    elif label_iter[i][3] == 1:
        Noisy =Noisy + 1
print('N:',N,'A',A,'O',O,'Noisy',Noisy)

dataN = np.zeros((4060,1,12,18286))
labelN = np.zeros((4060,4,1,1))
dataA = np.zeros((606,1,12,18286))
labelA = np.zeros((606,4,1,1))
dataO = np.zeros((1932,1,12,18286))
labelO = np.zeros((1932,4,1,1))
dataNoisy = np.zeros((223,1,12,18286))
labelNoisy = np.zeros((223,4,1,1))
K1,K2,K3,K4=[0,0,0,0]
for i in range(label_iter.shape[0]):
    if label_iter[i][0] == 1:
        dataN[K1]=data[i]
        labelN[K1]=label[i]
        K1 = K1+1
    elif label_iter[i][1] == 1:
        dataA[K2]=data[i]
        labelA[K2]=label[i]
        K2 = K2+1
    elif label_iter[i][2] == 1:
        dataO[K3] = data[i]
        labelO[K3] = label[i]
        K3 = K3 + 1
    elif label_iter[i][3] == 1:
        dataNoisy[K4]=data[i]
        labelNoisy[K4]=label[i]
        K4 = K4+1
f = h5py.File('./hdf5file/N_train.h5', 'w')
f.create_dataset('data', data=dataN)
f.create_dataset('label', data=labelN)
f.close()
f = h5py.File('./hdf5file/A_train.h5', 'w')
f.create_dataset('data', data=dataA)
f.create_dataset('label', data=labelA)
f.close()
f = h5py.File('./hdf5file/O_train.h5', 'w')
f.create_dataset('data', data=dataO)
f.create_dataset('label', data=labelO)
f.close()
f = h5py.File('./hdf5file/Noisy_train.h5', 'w')
f.create_dataset('data', data=dataNoisy)
f.create_dataset('label', data=labelNoisy)
f.close()