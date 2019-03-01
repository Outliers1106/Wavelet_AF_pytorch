import h5py
import numpy as np
# N: 4060 A 606 O 1932 Noisy 223
# 1015 x 4
# 151 151 152 152
# 483 483 483 483
# 55 55 56 57
#1704 1704 1706 1707
fN = h5py.File('./hdf5file/N_train.h5', 'r')
dataN = fN['data']
labelN = fN['label']
fA = h5py.File('./hdf5file/A_train.h5', 'r')
dataA = fA['data']
labelA = fA['label']
fO = h5py.File('./hdf5file/O_train.h5', 'r')
dataO = fO['data']
labelO = fO['label']
fNoisy = h5py.File('./hdf5file/Noisy_train.h5', 'r')
dataNoisy = fNoisy['data']
labelNoisy = fNoisy['label']

fold1 = np.zeros((1704,1,12,18286))
fold1 = fold1.astype(np.float32)
label1 = np.zeros((1704,4,1,1))
fold2 = np.zeros((1704,1,12,18286))
fold2 = fold2.astype(np.float32)
label2 = np.zeros((1704,4,1,1))
fold3 = np.zeros((1706,1,12,18286))
fold3 = fold3.astype(np.float32)
label3 = np.zeros((1706,4,1,1))
fold4 = np.zeros((1707,1,12,18286))
fold4 = fold4.astype(np.float32)
label4 = np.zeros((1707,4,1,1))
# write fold 1

print('writing fold1.......')
t=0
for i in range(1015):
    fold1[t] = dataN[i]
    label1[t] = labelN[i]
    t = t+1
for i in range(151):
    fold1[t] = dataA[i]
    label1[t] = labelA[i]
    t = t+1
for i in range(483):
    fold1[t] = dataO[i]
    label1[t] = labelO[i]
    t = t+1
for i in range(55):
    fold1[t] = dataNoisy[i]
    label1[t] = labelNoisy[i]
    t = t+1
f = h5py.File('./hdf5file/ecg_train_f1', 'w')
f.create_dataset('data', data=fold1)
f.create_dataset('label', data=label1)
f.close()
# write fold 2
print('write fold2.......')
t=0
for i in range(1015,1015+1015):
    fold2[t] = dataN[i]
    label2[t] = labelN[i]
    t = t+1
for i in range(151,151+151):
    fold2[t] = dataA[i]
    label2[t] = labelA[i]
    t = t+1
for i in range(483,483+483):
    fold2[t] = dataO[i]
    label2[t] = labelO[i]
    t = t+1
for i in range(55,55+55):
    fold2[t] = dataNoisy[i]
    label2[t] = labelNoisy[i]
    t = t+1
f = h5py.File('./hdf5file/ecg_train_f2', 'w')
f.create_dataset('data', data=fold2)
f.create_dataset('label', data=label2)
f.close()

# write fold 3
print('write fold3.........')
t=0
for i in range(1015+1015,1015+1015+1015):
    fold3[t] = dataN[i]
    label3[t] = labelN[i]
    t = t+1
for i in range(151+151,151+151+152):
    fold3[t] = dataA[i]
    label3[t] = labelA[i]
    t = t+1
for i in range(483+483,483+483+483):
    fold3[t] = dataO[i]
    label3[t] = labelO[i]
    t = t+1
for i in range(55+55,55+55+56):
    fold3[t] = dataNoisy[i]
    label3[t] = labelNoisy[i]
    t = t+1
f = h5py.File('./hdf5file/ecg_train_f3', 'w')
f.create_dataset('data', data=fold3)
f.create_dataset('label', data=label3)
f.close()

# write fold 4
print('write fold4............')
t=0
for i in range(1015+1015+1015,1015+1015+1015+1015):
    fold4[t] = dataN[i]
    label4[t] = labelN[i]
    t = t+1
for i in range(151+151+152,151+151+152+152):
    fold4[t] = dataA[i]
    label4[t] = labelA[i]
    t = t+1
for i in range(483+483+483,483+483+483+483):
    fold4[t] = dataO[i]
    label4[t] = labelO[i]
    t = t+1
for i in range(55+55+56,55+55+56+57):
    fold4[t] = dataNoisy[i]
    label4[t] = labelNoisy[i]
    t = t+1
f = h5py.File('./hdf5file/ecg_train_f4', 'w')
f.create_dataset('data', data=fold4)
f.create_dataset('label', data=label4)
f.close()
