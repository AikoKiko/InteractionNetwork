
# coding: utf-8

# In[1]:


import os
import setGPU
import numpy as np
from numpy import random
import h5py
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
from decimal import Decimal
import torch 
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch
from torch.autograd import Variable
import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from operator import itemgetter
import itertools
from sklearn.model_selection import train_test_split, StratifiedKFold


# In[2]:


file = h5py.File('/bigdata/shared/HLS4ML/jetImage.h5', 'r')
print(list(file.keys()))
sorted_pt_constituents = np.array(file.get('jetConstituentList'))
scaled_jets = file.get('jets')
targets = scaled_jets[:,-6:-1]


# In[3]:


def shuffle(a,b):
    iX = a.shape[1]
    iY = a.shape[2]
    b_shape = b.shape[1]
    a = a.reshape(a.shape[0], iX*iY)
    total = np.column_stack((a,b))
    random.seed(1)
    np.random.shuffle(total)
    a = total[:,:iX*iY]
    b = total[:,iX*iY:iX*iY+b_shape]
    a = a.reshape(a.shape[0],iX, iY)
    return a,b


# In[4]:


sorted_pt_constituents, targets = shuffle(sorted_pt_constituents, targets)
y_train =np.argmax(targets, axis=1)


# In[5]:


fraction = input("Please enter what fraction of dataset you want: ")
print("You entered " + str(fraction))
fraction = float(fraction)


# In[6]:


num = targets.shape[0]
split_dataset = int(np.floor(fraction * num))
sorted_pt_constituents = sorted_pt_constituents[:split_dataset,:,:]
targets =targets[:split_dataset]
print(sorted_pt_constituents.shape)
print(targets.shape)


# In[7]:


No = 188
P=16
Nr = int(No*(No-1)/2)


# In[9]:


connection_list = [i for i in itertools.product(range(187), range(188)) if i[0]!=i[1] and i[0]<i[1]]


# In[ ]:


RR = np.array([])
for arr in sorted_pt_constituents:
    Rr = np.zeros((Nr, No))
    for i, (r, s) in enumerate(connection_list):
        if arr[r, 0]==0 or arr[s,0] == 0:
            Rr[i, r] = 0
            Rr[i, s] = 0
        else:
            Rr[i, r] = 1
            Rr[i, s] = 1
    RR = np.concatenate((RR,Rr), axis=0) if RR.size else Rr         
print(RR)


# In[ ]:


RR = np.transpose(RR)


# In[8]:


def load_data_kfold(k,sorted_pt_constituents,RR, y_train):
        
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(sorted_pt_constituents, RR, y_train))
    
    return folds


# In[9]:


k = 5
folds = load_data_kfold(k,sorted_pt_constituents,RR,y_train)


# In[5]:


import torch.nn.functional as F
class GraphNet(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden):
        super(GraphNet, self).__init__()
        self.hidden = hidden
        self.P = params
        self.N = n_constituents
        self.Dr = 0
        self.De = 5
        self.Dx = 0
        self.Do = 6
        self.Nr=Nr
        self.n_targets = n_targets
        self.fr1 = nn.Linear(self.P + self.Dr, hidden)
        self.fr2 = nn.Linear(hidden, int(hidden/2))
        self.fr3 = nn.Linear(int(hidden/2), self.De)
        self.fo1 = nn.Linear(self.P + self.Dx + self.De, hidden)
        self.fo2 = nn.Linear(hidden, int(hidden/2))
        self.fo3 = nn.Linear(int(hidden/2), self.Do)
        self.fc1 = nn.Linear(self.Do * self.N, hidden)
        self.fc2 = nn.Linear(hidden, int(hidden/2))
        self.fc3 = nn.Linear(int(hidden/2), self.n_targets)
    
    def forward(self, x, RR):
        x=torch.transpose(x, 1, 2).contiguous()
        Orr = self.tmul(x, self.RR)
        B = torch.transpose(Orr, 1, 2).contiguous()
        ### First MLP ###
        B = nn.functional.tanh(self.fr1(B.view(-1, self.P + self.Dr)))
        B = nn.functional.tanh(self.fr2(B))
        E = nn.functional.tanh(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        C = torch.cat([x, Ebar], 1)
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        C = nn.functional.tanh(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
        C = nn.functional.tanh(self.fo2(C))
        O = nn.functional.tanh(self.fo3(C).view(-1, self.N, self.Do))
        O = torch.sum(O,  dim=1)
        del C
        ### Classification MLP ###
        N = nn.functional.tanh(self.fc1(O.view(-1, self.Do)))
        del O
        N = nn.functional.tanh(self.fc2(N))
        N = self.fc3(N)
        return N
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])


# In[11]:


batch_size=128
epochs =200


# In[12]:


class EventDataset(Dataset):
    def __init__(self, constituents, targets,
                 constituents_name = ['j1_pt', 'j1_etarel','j1_phirel']
                ):
        self.constituents = torch.from_numpy(constituents)
        self.targets = torch.from_numpy(targets)
        self.constituents_name = constituents_name
    def __len__(self):
        return self.constituents.shape[0]
    def __getitem__(self,idx):
        return self.constituents[idx], self.targets[idx]


# In[17]:


f= open('IN_fr_dataset_%f.txt' % fraction, 'w')


# In[14]:


class EarlyStopping(object):
    """
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self, 
                 monitor='val_loss',
                 min_delta=0,
                 patience=30):
        super(EarlyStopping, self).__init__()
        print("This is my patience {}".format(patience))
        f.write("This is my patience {}\n".format(patience)) 
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e-15
        self.stopped_epoch = 0
        self.stop_training= False
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_loss = 1e15

    def on_epoch_end(self, epoch, current_loss):
        print("This is current loss {}".format(current_loss))
        f.write("This is current loss {}\n".format(current_loss))
        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                print("This is best loss {}".format(self.best_loss))
                f.write("This is best loss {}\n".format(self.best_loss))
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    self.stop_training = True
                self.wait += 1
            return  self.stop_training
        
    def on_train_end(self):
        if self.stopped_epoch > 0:
            print('\nTerminated Training for Early Stopping at Epoch %04i' % 
                (self.stopped_epoch))


# In[15]:


def train_model(data_loader,num_epochs, model, criterion, optimizer,scheduler,volatile=False):

    best_model = model.state_dict()
    best_acc = 0.0
    train_losses ,val_losses = [],[]
    Early_Stopping = EarlyStopping(patience=30)
    Early_Stopping.on_train_begin()
    breakdown = False
    for epoch in range(num_epochs):
        if breakdown:
            print("Early Stopped")
            f.write('Early Stopped\n') 
            break
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        f.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        print('_' * 10)
        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode
                volatile=True
            running_loss = 0.0
            running_corrects = 0

        # Iterate over data.
            for batch_idx, (x_data, y_data) in enumerate(data_loader[phase]):
                x_data, y_data = Variable(x_data.cuda().type(torch.cuda.FloatTensor),volatile),Variable(y_data.cuda().type(torch.cuda.LongTensor))
                if phase == 'training':
                    optimizer.zero_grad()
                # forwardgyg
                outputs = model(x_data)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, y_data)
                # backward + optimize only if in training phase
                if phase == 'training':
                    loss.backward()
                    optimizer.step()
                
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == y_data.data)
                #print("I finished %d batch" % batch_idx)
            
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = 100. * running_corrects / len(data_loader[phase].dataset)
            if phase == 'training':
                train_losses.append(epoch_loss)
            else:
                scheduler.step(epoch_loss)
                val_losses.append(epoch_loss)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                print('Saving..')
                state = {
                        'net': model, #.module if use_cuda else net,
                        'epoch': epoch,
                        'best_acc':epoch_acc,
                        'train_loss':train_losses,
                        'val_loss':val_losses,
                        }
                if not os.path.isdir('checkpoint4'):
                    os.mkdir('checkpoint4')
                torch.save(state, './checkpoint4/IN.t7')
                best_acc = epoch_acc
                best_model = model.state_dict()
            if phase == 'validation':
                breakdown = Early_Stopping.on_epoch_end(epoch,round(epoch_loss,4))
                
            print()

    
    print('Best val Acc: {:4f}'.format(best_acc))
    f.write('Best val Acc: {:4f}\n'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model)
    return model,train_losses ,val_losses


# In[16]:


for j, (train_idx, val_idx) in enumerate(folds):
        print('\nFold ',j)
        f.write('\nFold {}'.format(j))
        X_train_cv = sorted_pt_constituents[train_idx]
        y_train_cv = y_train[train_idx]
        X_valid_cv = sorted_pt_constituents[val_idx]
        y_valid_cv= y_train[val_idx]
        train = EventDataset(X_train_cv,y_train_cv)  
        valid = EventDataset(X_valid_cv,y_valid_cv)  
        train_loader = DataLoader(dataset = train, batch_size = batch_size, shuffle = True, num_workers = 4)
        valid_loader = DataLoader(dataset = valid, batch_size = batch_size, shuffle = True, num_workers = 4) 
        data_loader = {"training" :train_loader, "validation" : valid_loader}
        model = GraphNet(n_constituents=188,n_targets=5,params=16, hidden=10)
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='min',factor=0.5,patience=10)
        model_output,train_losses,val_losses = train_model(data_loader,epochs,model,criterion,optimizer,scheduler)
        np.save('train_losses of folder {} and fraction {}'.format(j, fraction),train_losses)
        np.save('val_losees of folder {} and fraction {}'.format(j, fraction), val_losses)
        print('Saving after {} fold..'.format(j))
        state = {
                    'net': model_output, #.module if use_cuda else net,
                    'train_loss':train_losses,
                    'val_loss':val_losses
                        }
        if not os.path.isdir('folder_checkpoint_4'):
            os.mkdir('folder_checkpoint_4')
        torch.save(state, './folder_checkpoint_4/folder_IN_.t7')


# In[ ]:


f.close()

