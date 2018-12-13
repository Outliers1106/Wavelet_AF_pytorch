import torch.nn as nn
import torch
import h5py
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
GAMMA = 0.1
STEP_SIZE = 5000
MAX_ITER = 30000
BATCH_SIZE = 100

#train

#net  struction
class mynet(nn.Module):
    def __init__(self):
        super(mynet,self).__init__()
        #input 1x12x18286
        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=1,out_channels=32,
                               kernel_size=(3,11),stride=(1,4)),
                     #32x10x4565
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=(2,3),stride=(2,3)
                     #32x5x1521
                     )
        )
        self.conv2 = nn.Sequential(
                     nn.Conv2d(in_channels=32,out_channels=32,
                               kernel_size=(2,11),stride=(1,4)),
                     #32x4x378
                     nn.MaxPool2d(kernel_size=(2,3),stride=(2,3)
                     #32x2x126
                     )
        )
        self.fc1 = nn.Linear(32*2*126,100)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100,4)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)#zhan kai
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x


def OnehotToScalar(label):#one-hot to biaoliang for cross Entropy 
    label_new = torch.zeros(label.size(0))
    for i in range(label.size(0)):
        if(label[i][0]==1):
            label_new[i] = 0
        elif(label[i][1]==1):
            label_new[i] = 1
        elif(label[i][2]==1):
            label_new[i] = 2
        else:
            label_new[i] = 3
    return label_new

def restore_parameters():
    net = mynet()
    net.load_state_dict(torch.load('mynet_params.pkl'))
    return net

def adjust_learning_rate(optimizer, iters = 0):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * GAMMA**(np.floor(iters/STEP_SIZE))
        
def test_accuracy(pred_y,test_label):
    sumN=0,sumA=0,sumO=0,sumNoisy=0
    sumN1=0,sumA1=0,sumO1=0,sumNoisy=0
    for i in range(test_label.size):
        if test_label[i] = 0:
            sumN = sumN + 1
        elif test_label[i] = 1:
            sumA = sumA + 1
        elif test_label[i] = 2:
            sumO = sumO + 1
        else:
            sumNoisy = sumNoisy + 1
    for i in range(test_label.size):
        if test_label[i]==0 and pred_y[i]==0:
            sumN1 = sumN1 + 1
        elif test_label[i]==1 and pred_y[i]==1:
            sumA1 = sumA1 + 1
        elif test_label[i]==2 and pred_y[i]==2:
            sumO1 = sumO1 + 1
        elif test_label[i]==3 and pred_y[i]==3:
            sumNoisy1 = sumNoisy1 + 1
    accuracyN = sumN/sumN1
    accuracyA = sumA/sumA1
    accuracyO = sumO/sumO1
    accuracyNoisy = sumNoisy/sumNoisy1
    return accuracyN,accuracyA,accuracyO,accuracyNoisy

if __name__ =='__main__':
    #load train data and train label
    f = h5py.File('ecg_traindata.h5','r')
    data = f['data']
    label = f['label']
    #change hdf5data to numpy
    data = data[:][:][:][:]
    label = label[:][:][:][:]
    f.close()
    #change data type numpy to torch
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    label = label.squeeze(3)#nx4x1x1 ->nx4x1
    label = label.squeeze(2)#nx4x1 ->nx4
    
    label = OnehotToScalar(label)
    
    torch_dataset = Data.TensorDataset(data_tensor=data, target_tensor=label)
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 2
    )
    '''
    load test data and test label
    '''
    '''
    f = h5py.File('ecg_testdata.h5','r')
    test_data = f['data']
    test_label = f['label']
    test_data = test_data[:][:][:][:]
    test_label = test_label[:][:][:][:]
    f.close()
    test_label = test_label.squeeze(3)#nx4x1x1 ->nx4x1
    test_label = test_label.squeeze(2)#nx4x1 ->nx4
    test_label = OnehotToScalar(test_label)
    test_data = Variable(test_data)
    #test_label = Variable(test_label)
    '''
    '''
    train
    '''
    mynet = mynet()
    if torch.cuda.is_available():
        mynet.cuda()
    opt_SGD = torch.optim.SGD([
        {'params':mynet.parameters()}
        ],lr=LR,momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)
    loss_func = torch.nn.CrossEntropyLoss()
    
    for epoch in range(MAX_ITER):
        print('Epoch:',epoch)
        adjust_learning_rate(opt_SGD,epoch)
        for step,(b_x,b_y) in enumerate(loader):
            b_x = Variable(b_x)
            b_y = Variable(b_y)
            output = mynet(b_x)
            b_y = b_y.long()
            loss = loss_func(output,b_y)
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()
            if(step % 30 == 0):
                test_output = mynet(test_data)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                A,O,N,Noisy = test_accuracy(pred_y,test_label)
                accuracy = float((pred_y == test_label).sum()) / float(test_label.size(0)) 
                print('Epoch:',epoch,'|step:',step,'|loss:%.4f',loss.data[0],'test_accuracy:%.2f',accuracy,
                      'A:%.2f',A,'N:%.2f',N,'O:%.2f',O,'Noisy:%.2f',Noisy)
                torch.save(mynet.state_dict(), './model/mynet_'+Epoch+'_'+step+'params.pkl')   #save parameters of net 
    
    