import torch.nn as nn
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
from PIL import Image
import h5py
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import random


LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
GAMMA = 0.1
STEP_SIZE = 200
MAX_ITER = 500
BATCH_SIZE = 100

class Custom_dataset(torch.utils.data.Dataset):
    def __len__(self):
        return self.len

    def __init__(self,path:str):
        super().__init__()
        self.path = path
        f1 = h5py.File(self.path+'ecg_testdata.h5','r')
        f2 = h5py.File(self.path+'ecg_train_f1','r')
        f3 = h5py.File(self.path+'ecg_train_f2','r')
        f4 = h5py.File(self.path+'ecg_train_f4','r')
        self.data1 = f1['data']
        self.data2 = f2['data']
        self.data3 = f3['data']
        self.data4 = f4['data']
        self.label1 = OnehotToScalar(f1['label'][:].squeeze(3).squeeze(2))#nx4x1x1->nx4
        self.label2 = OnehotToScalar(f2['label'][:].squeeze(3).squeeze(2))
        self.label3 = OnehotToScalar(f3['label'][:].squeeze(3).squeeze(2))
        self.label4 = OnehotToScalar(f4['label'][:].squeeze(3).squeeze(2))
        self.len = self.label1.shape[0] + self.label2.shape[0] +self.label3.shape[0]+self.label4.shape[0]

    def __getitem__(self,index:int):
        len1 = self.label1.shape[0]
        len2 = self.label2.shape[0]
        len3 = self.label3.shape[0]
        len4 = self.label4.shape[0]
        data,label=[0,0]
        if index < len1:
            data = self.data1[index]
            label = self.label1[index]
        elif index >=len1 and index < len1+len2:
            data = self.data2[index-len1]
            label = self.label2[index-len1]
        elif index >=(len1+len2) and index < (len1+len2+len3):
            data = self.data3[index-len1-len2]
            label = self.label3[index-len1-len2]
        elif index >=(len1+len2+len3) and index < (len1+len2+len3+len4):
            data = self.data4[index-len1-len2-len3]
            label = self.label4[index-len1-len2-len3]
        else:
            raise RuntimeError('index is wrong!!!')
        print(data.shape)
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(np.array(label))
        return data,label

class Custom_dataset_test(torch.utils.data.Dataset):
    def __len__(self):
        return self.len

    def __init__(self,path:str):
        super().__init__()
        self.path = path
        f1 = h5py.File(self.path+'ecg_train_f3','r')
        self.data1 = f1['data']
        self.label1 = OnehotToScalar(f1['label'][:].squeeze(3).squeeze(2))
        self.len = self.label1.shape[0]

    def __getitem__(self,index:int):
        data = self.data1[index]
        label = self.label1[index]
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(np.array(label))
        return data,label

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
                     #32x10x4569
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=(2,3),stride=(2,3)
                     #32x5x1521
                     #32x5x1523
                     )
        )
        self.conv2 = nn.Sequential(
                     nn.Conv2d(in_channels=32,out_channels=32,
                               kernel_size=(2,11),stride=(1,4)),
                     #32x4x378
                     #32x4x379
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
        #if torch.cuda.is_available():
        #    x = x.cuda()
        return x


def OnehotToScalar(label):#one-hot to biaoliang for cross Entropy 
    label_new = torch.zeros(label.shape[0])
    for i in range(label.shape[0]):
        if(label[i][0]==1):
            label_new[i] = 0
        elif(label[i][1]==1):
            label_new[i] = 1
        elif(label[i][2]==1):
            label_new[i] = 2
        else:
            label_new[i] = 3
    return label_new

def restore_parameters(name:str):
    net = mynet()
    net.load_state_dict(torch.load('./modelAll/'+name))
    return net

def adjust_learning_rate(optimizer, iters = 0):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * GAMMA**(np.floor(iters/STEP_SIZE))
'''
def get_random_test(test_data,test_label):
    index = random.sample(range(0,test_label.shape[0]),100)
    return test_data[index],test_label[index]
'''

def test_accuracy(pred_y,test_label):
    sumN,sumA,sumO,sumNoisy=[0,0,0,0]
    sumN1,sumA1,sumO1,sumNoisy1=[0,0,0,0]
    for i in range(test_label.size(0)):
        if test_label[i] == 0:
            sumN = sumN + 1
        elif test_label[i] == 1:
            sumA = sumA + 1
        elif test_label[i] == 2:
            sumO = sumO + 1
        else:
            sumNoisy = sumNoisy + 1
    for i in range(test_label.size(0)):
        if test_label[i]==0 and pred_y[i]==0:
            sumN1 = sumN1 + 1
        elif test_label[i]==1 and pred_y[i]==1:
            sumA1 = sumA1 + 1
        elif test_label[i]==2 and pred_y[i]==2:
            sumO1 = sumO1 + 1
        elif test_label[i]==3 and pred_y[i]==3:
            sumNoisy1 = sumNoisy1 + 1
    return sumN,sumN1,sumA,sumA1,sumO,sumO1,sumNoisy,sumNoisy1
    accuracyN = sumN1/(sumN+0.0001)#avoid zero division
    accuracyA = sumA1/(sumA+0.0001)
    accuracyO = sumO1/(sumO+0.0001)
    accuracyNoisy = sumNoisy1/(sumNoisy+0.0001)
    return accuracyN,accuracyA,accuracyO,accuracyNoisy


def F1score(pred_y,label):
    n = label.shape[0]
    AA = np.zeros((4,4))
    for i in range(n):
        if label[i] == 0:
            if pred_y[i] == 0:
                AA[0][0] = AA[0][0]+1
            elif pred_y[i] ==1:
                AA[0][1] = AA[0][1]+1
            elif pred_y[i] ==2:
                AA[0][2] = AA[0][2]+1
            elif pred_y[i] ==3:
                AA[0][3] = AA[0][3]+1
        elif label[i] == 1:
            if pred_y[i] == 0:
                AA[1][0] = AA[1][0]+1
            elif pred_y[i] ==1:
                AA[1][1] = AA[1][1]+1
            elif pred_y[i] ==2:
                AA[1][2] = AA[1][2]+1
            elif pred_y[i] ==3:
                AA[1][3] = AA[1][3]+1
        elif label[i] == 2:
            if pred_y[i] == 0:
                AA[2][0] = AA[2][0]+1
            elif pred_y[i] ==1:
                AA[2][1] = AA[2][1]+1
            elif pred_y[i] ==2:
                AA[2][2] = AA[2][2]+1
            elif pred_y[i] ==3:
                AA[2][3] = AA[2][3]+1
        elif label[i] == 3:
            if pred_y[i] == 0:
                AA[3][0] = AA[3][0]+1
            elif pred_y[i] ==1:
                AA[3][1] = AA[3][1]+1
            elif pred_y[i] ==2:
                AA[3][2] = AA[3][2]+1
            elif pred_y[i] ==3:
                AA[3][3] = AA[3][3]+1
    return AA
    F1n=2*AA[0][0]/(sum(AA[0][:])+sum(AA[:][0]))
    F1a=2*AA[1][1]/(sum(AA[1][:])+sum(AA[:][1]))
    F1o=2*AA[2][2]/(sum(AA[2][:])+sum(AA[:][2]))
    F1p=2*AA[3][3]/(sum(AA[3][:])+sum(AA[:][3]))
    F1=(F1n+F1a+F1o+F1p)/4
    print('f1n %1.4f,'%F1n,'f1a %1.4f,'%F1a,'flo %1.4f,'%F1o,'f1p %1.4f.'%F1p)
    print('f1 overall %1.4f'%F1)

def testdata(loader_test, mynet,epoch):
    mynet.eval()
    AA =np.zeros((4,4))
    sumN,sumN1,sumA,sumA1,sumO,sumO1,sumNoisy,sumNoisy1=[0,0,0,0,0,0,0,0]
    for step, (b_x, b_y) in enumerate(loader_test):
        b_x = Variable(b_x).cuda()
        b_y = Variable(b_y).cuda()
        output = mynet(b_x)
        b_y = b_y.long()
        pred_y = torch.max(output, 1)[1].data.squeeze().cpu().numpy()
        if epoch % 20==0:
            sumN_, sumN1_, sumA_, sumA1_, sumO_, sumO1_, sumNoisy_, sumNoisy1_ = test_accuracy(pred_y,b_y)
            sumN, sumN1, sumA, sumA1, sumO, sumO1, sumNoisy, sumNoisy1=[sumN+sumN_,sumN1+sumN1_,sumA+sumA_,sumA1+sumA1_,sumO+sumO_,sumO1+sumO1_,sumNoisy+sumNoisy_,sumNoisy1+sumNoisy1_]
        AA1 = F1score(pred_y, b_y)
        AA = AA +AA1
    F1n = 2 * AA[0][0] / (sum(AA)[0] + sum(AA.transpose())[0])
    F1a = 2 * AA[1][1] / (sum(AA)[1] + sum(AA.transpose())[1])
    F1o = 2 * AA[2][2] / (sum(AA)[2] + sum(AA.transpose())[2])
    F1p = 2 * AA[3][3] / (sum(AA)[3] + sum(AA.transpose())[3])
    F1 = (F1n + F1a + F1o + F1p) / 4
    print('f1n %1.4f,' % F1n, 'f1a %1.4f,' % F1a, 'flo %1.4f,' % F1o, 'f1p %1.4f.' % F1p)
    print('f1 overall %1.4f' % F1)
    if epoch % 20==0:
        print('accuracy:N:%1.4f, '%float(sumN1/sumN),'A:%1.4f,'%float(sumA1/sumA),'O:%1.4f,'%float(sumO1/sumO),'Noisy:%1.4f'%float(sumNoisy1/sumNoisy))

    return F1,F1n,F1a,F1o,F1p
    #print('test accuary: N:', float(N1 / N), ' A:', float(A1 / A), ' O:', float(O1 / O), ' Noisy:',float(Noisy1 / Noisy))
    #print('N1:%d/N:%d',N1,N,
# load test
testset = Custom_dataset_test(path='./hdf5file/')
loader_test = Data.DataLoader(
    dataset=testset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
# load train
dataset = Custom_dataset(path='./hdf5file/')
loader = Data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


loss_func = torch.nn.CrossEntropyLoss()
if __name__ =='__main__':
    '''
    train
    '''
    #mynet1 = restore_parameters('model1/mynet_170_params.pkl')
    mynet1 = mynet()
    if torch.cuda.is_available():
        mynet1.cuda()

    opt_SGD = torch.optim.SGD([
        {'params': mynet1.parameters()}
    ], lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    loss_save =[]
    F1_save = [[],[],[],[],[]]

    
    for epoch in range(MAX_ITER):
        mynet1.train()
        print('Epoch:',epoch)
        adjust_learning_rate(opt_SGD,epoch)

        for step,(b_x,b_y) in enumerate(loader):
            b_x = Variable(b_x).cuda()
            b_y = Variable(b_y).cuda()
            output = mynet1(b_x)
            b_y = b_y.long()
            loss = loss_func(output,b_y)
            loss_save.append(loss.data[0])
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()
            pred_y = torch.max(output, 1)[1].data.squeeze().cpu().numpy()
            b_y = b_y.cpu().numpy()
            accuracy = float((pred_y == b_y).sum()) / float(b_y.size)
            print('Epoch:', epoch, '|step:', step, '|loss:%.4f' % loss.data[0], 'train_accuracy:%.2f' % accuracy)

        scoreAll,scoreN,scoreA,scoreO,scoreP = testdata(loader_test, mynet1,epoch)
        F1_save[0].append(scoreAll)
        F1_save[1].append(scoreN)
        F1_save[2].append( scoreA)
        F1_save[3].append( scoreO)
        F1_save[4].append( scoreP)
        torch.save(mynet1.state_dict(), './modelAll/model_test_fold3/mynet_' + str(epoch) + '_' + 'params.pkl')  # save parameters of net
    plt.plot(loss_save)
    plt.savefig('./lossimage/lossimage_test_fold3/loss.png')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.close()
    labels=['F1','F1n','F1a','F1o','F1p']
    for i,l_his in enumerate(F1_save):
         plt.plot(l_his,label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.savefig('./lossimage/lossimage_test_fold3/F1.png')
    plt.close()
