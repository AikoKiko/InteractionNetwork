
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
torch.cuda.manual_seed_all(1)
import numpy.ma as ma
from sklearn.model_selection import train_test_split


# In[2]:

batch_size=128
epochs =200
test_size = 0.2
valid_size = 0.25


# In[3]:

from glob import glob
from sklearn.utils import shuffle

class EventImage(Dataset):

    def check_data(self, file_names):
        #done
        '''Count the number of events in each file and mark the threshold 
        boundaries between adjacent indices coming from 2 different files'''
        num_data = 0
        thresholds = [0]
        for in_file_name in file_names:
            h5_file = h5py.File( in_file_name, 'r' )
            X = h5_file[self.const[0]]
            if hasattr(X, 'keys'):
                num_data += len(X[X.keys()[0]])
                thresholds.append(num_data)
            else:
                num_data += len(X)
                thresholds.append(num_data)
            h5_file.close()
        return (num_data, thresholds) # threshholds = [0,20,40,60,80,100 ....], num_data = total data size

    def __init__(self, dir_name, const = ['RA', 'RR', 'RS', 'jetConstituentList','jets']):
        #done
        self.const = const
        self.file_names = dir_name
        self.num_data, self.thresholds = self.check_data(self.file_names)

    def is_numpy_array(self, data):
        return isinstance(data, np.ndarray)

    def get_num_samples(self, data):
        """Input: dataset consisting of a numpy array or list of numpy arrays.
            Output: number of samples in the dataset"""
        if self.is_numpy_array(data):
            return len(data)
        else:
            return len(data[0])

    def load_data(self, in_file_name):
        #done
        """Loads numpy arrays from H5 file.
            If the features/labels groups contain more than one dataset,
            we load them all, alphabetically by key."""
        h5_file = h5py.File( in_file_name, 'r' )
        RA = np.array(self.load_hdf5_data(h5_file[self.const[0]] ))
        RR = np.array(self.load_hdf5_data(h5_file[self.const[1]] ))
        RS = np.array(self.load_hdf5_data(h5_file[self.const[2]] ))
        jetConstituentList = np.array(self.load_hdf5_data(h5_file[self.const[3]]))
        target = np.array(self.load_hdf5_data(h5_file[self.const[4]])[:,-6:-1])
        h5_file.close()
        RA,RR,RS,jetConstituentList,target = shuffle(RA,RR,RS,jetConstituentList,target,random_state=0)
        return RA,RR,RS,jetConstituentList,target
    
    def load_hdf5_data(self, data):
        #done
        """Returns a numpy array or (possibly nested) list of numpy arrays
            corresponding to the group structure of the input HDF5 data.
            If a group has more than one key, we give its datasets alphabetically by key"""
        if hasattr(data, 'keys'):
            out = [ self.load_hdf5_data( data[key] ) for key in sorted(data.keys()) ]
        else:
            out = data[:]
        return out

    def get_data(self, data, idx):
        #done
        """Input: a numpy array or list of numpy arrays.
            Gets elements at idx for each array"""
        if self.is_numpy_array(data):
            return data[idx]
        else:
            return [arr[idx] for arr in data]

    def get_index(self, idx):
        #done
        """Translate the global index (idx) into local indexes,
        including file index and event index of that file"""
        file_index = next(i for i,v in enumerate(self.thresholds) if v > idx)
        file_index -= 1
        event_index = idx - self.thresholds[file_index]
        return file_index, event_index # return file number where the event is located and which event to consider in this file;

    def get_thresholds(self):
        return self.thresholds
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        file_index, event_index = self.get_index(idx) # done 
        RA,RR,RS,jetConstituentList,target = self.load_data(self.file_names[file_index]) # done
        return  self.get_data(RA, event_index), self.get_data(RR, event_index),self.get_data(RS, event_index),self.get_data(jetConstituentList, event_index), np.argmax(self.get_data(target, event_index))

all_files = glob('/bigdata/shared/HLS4ML/NEW/AS/CC/CCC/*.h5')
shuffle(all_files)
testing_data = all_files[:int(test_size*len(all_files))]
training_data =  all_files[int(test_size*len(all_files)):]
training_data, validation_data = training_data[int(valid_size*len(training_data)):],training_data[:int(valid_size*len(training_data))]


# In[4]:

train_set = EventImage(training_data)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_set = EventImage(validation_data)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True , num_workers=0)
test_set = EventImage(testing_data)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True , num_workers=0)
data_loader = {"training" :train_loader, "validation" : val_loader}


# In[5]:

import torch.nn.functional as F
class GraphNet(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden):
        super(GraphNet, self).__init__()
        self.hidden = hidden
        self.P = params
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Dr = 3
        self.De = 5
        self.Dx = 0
        self.Do = 6
        self.n_targets = n_targets
        self.fr1 = nn.Linear(2 * self.P + self.Dr, hidden)
        self.fr2 = nn.Linear(hidden, int(hidden/2))
        self.fr3 = nn.Linear(int(hidden/2), self.De)
        self.fo1 = nn.Linear(self.P + self.Dx + self.De, hidden)
        self.fo2 = nn.Linear(hidden, int(hidden/2))
        self.fo3 = nn.Linear(int(hidden/2), self.Do)
        self.fc1 = nn.Linear(self.Do * self.N, hidden)
        self.fc2 = nn.Linear(hidden, int(hidden/2))
        self.fc3 = nn.Linear(int(hidden/2), self.n_targets)
    def forward(self, x, RR, RS, RA):
        x=torch.transpose(x, 1, 2).contiguous()
        Orr = self.tmul(x, RR)
        Ors = self.tmul(x, RS)
        B = torch.cat([Orr, Ors], 1)
        B = torch.cat([B, RA], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
        B = nn.functional.relu(self.fr2(B))
        E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(RR, 1, 2).contiguous())
        del E
        C = torch.cat([x, Ebar], 1)
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
        C = nn.functional.relu(self.fo2(C))
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C
        ### Classification MLP ###
        N = nn.functional.relu(self.fc1(O.view(-1, self.Do * self.N)))
        del O
        N = nn.functional.relu(self.fc2(N))
        N = self.fc3(N)
        return N
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        XY = np.array([])
        for e, Rr in zip(x,y):
            xy = torch.mm(e, Rr).view(-1, e.shape[0], Rr.shape[1])
            XY = np.concatenate((XY,xy), axis=0) if XY.size else xy
        return Variable(torch.from_numpy(XY)).cuda()

model = GraphNet(n_constituents=188,n_targets=5,params=16, hidden=10)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=10e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='min',factor=0.5,patience=10)


# In[6]:

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
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training= False
    def on_train_begin(self, logs=None):
        self.wait = 0
    def on_epoch_end(self, epoch, val_loss):
        if val_loss is None:
            pass
        else:
            if len(val_loss) ==10:
                self.best_loss = np.mean(val_loss)
            if np.mean(val_loss[-10:]) < self.best_loss:
                self.best_loss = np.mean(val_loss[-10:])
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


# In[7]:

def train_model(num_epochs, model, criterion, optimizer,scheduler,volatile=False):
    best_model = model.state_dict()
    best_acc = 0.0
    train_losses ,val_losses = [],[]
    Early_Stopping = EarlyStopping(patience=20)
    Early_Stopping.on_train_begin()
    breakdown = False
    p=10
    for epoch in range(num_epochs):
        if breakdown:
            print("Early Stopped")
            f.write('Early Stopped\n') 
            break
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         f.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
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
            for batch_idx, (RA, RR, RS, jetConstituentList,target) in enumerate(data_loader[phase]):
                RA = Variable(RA,volatile).float().cuda()
                RS = Variable(RS,volatile).float().cuda()
                RR = Variable(RR,volatile).float().cuda()
                jetConstituentList = Variable(jetConstituentList,volatile).float().cuda()
                target = Variable(target).long().cuda()
                if phase == 'training':
                    optimizer.zero_grad()
                # forwarding
                outputs = model(jetConstituentList, RR, RS, RA)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, target)
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
#             f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(
#                 phase, epoch_loss, epoch_acc))

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
            if phase == 'validation' and epoch==p-1:
                breakdown = Early_Stopping.on_epoch_end(epoch, val_losses)
                p += p
                print()
    print('Best val Acc: {:4f}'.format(best_acc))
#     f.write('Best val Acc: {:4f}\n'.format(best_acc))
#     model.load_state_dict(best_model)
    return model,train_losses ,val_losses


# In[8]:

# f= open('IN_Aidanchan_%f.txt', 'w')


# In[ ]:

model_output,train_losses,val_losses = train_model(epochs,model,criterion,optimizer,scheduler)


# In[ ]:

# predicted = np.zeros((19661, 5))
# test_loss = 0
# test_correct = 0
# for batch_idx, (x_data,B,RR, y_data) in enumerate(test_loader):
#     x_data, y_data, B = Variable(x_data,volatile=True).float().cuda(),Variable(y_data).long().cuda(),Variable(B).float().cuda()
#     RR = Variable(RR,volatile=True).float().cuda()
#     model_out = model_output(x_data,B,RR)
#     beg = batch_size*batch_idx
#     end = min((batch_idx+1)*batch_size, 19661)
#     predicted[beg:end] = F.softmax(model_out).data.cpu().numpy()     
#     test_loss += criterion(model_out, y_data).data[0]
#     _, preds = torch.max(model_out.data, 1)  
#     test_correct += preds.eq(y_data.data).sum()

# test_loss /= len(test_loader.dataset)
# test_acc = 100. * test_correct/ len(test_loader.dataset)
# print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, test_correct, len(test_loader.dataset),
#         test_acc))
# f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, test_correct, len(test_loader.dataset),
#         test_acc))


# In[ ]:

# np.save('Aidanchan_train_losses',train_losses)
# np.save('Aidanchan_val_losses', val_losses)
# np.save('Aidanchan_predicted', predicted)
# np.save('Aidanchan_y_test', y_test)


# In[ ]:

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import OneHotEncoder
# labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
# predict_test = predicted
# enc = OneHotEncoder(5, sparse = False)
# labels_val = enc.fit_transform(y_test.reshape((4,1)))
# df = pd.DataFrame()
# fpr = {}
# tpr = {}
# auc1 = {}

# plt.figure()
# for i, label in enumerate(labels):
#         df[label] = labels_val[:,i]
#         df[label + '_pred_IN'] = predict_test[:,i]

#         fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred_IN'])

#         auc1[label] = auc(fpr[label], tpr[label])

#         plt.plot(tpr[label],fpr[label],label='%s tagger, auc = %.1f%%'%(label,auc1[label]*100.))
# df.to_csv("IN.csv", sep='\t')
# plt.semilogy()
# plt.xlabel("sig. efficiency")
# plt.ylabel("bkg. mistag rate")
# plt.ylim(0.0001,1)
# plt.grid(True)
# plt.legend(loc='upper left')
# #plt.savefig('%s/ROC.pdf'%(options.outputDir))
# plt.show()


# In[ ]:

# for j, (train_idx, val_idx) in enumerate(folds):
#         print('\n Fold ',j)
# #         f.write('\nFold {}'.format(j))
#         X_train_cv = sorted_pt_constituentsnp[train_idx]
#         y_train_cv = y_train[train_idx]
#         X_valid_cv = sorted_pt_constituentsnp[val_idx]
#         y_valid_cv= y_train[val_idx]
#         B_train_cv = B[train_idx]
#         B_valid_cv = B[val_idx]
#         R_train_cv = RR[train_idx]
#         R_valid_cv = RR[val_idx]
#         train = EventDataset(X_train_cv,B_train_cv,R_train_cv,y_train_cv)  
#         valid = EventDataset(X_valid_cv,B_valid_cv,R_valid_cv,y_valid_cv)  
#         train_loader = DataLoader(dataset = train, batch_size = batch_size, shuffle = True, num_workers = 4)
#         valid_loader = DataLoader(dataset = valid, batch_size = batch_size, shuffle = True, num_workers = 4)
#         data_loader = {"training" :train_loader, "validation" : valid_loader}
#         model = GraphNet(n_constituents=5,n_targets=5,params=16, hidden=10)
#         model = nn.DataParallel(model.cuda())
#         optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
#         criterion = nn.CrossEntropyLoss()
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='min',factor=0.5,patience=10)
#         model_output,train_losses,val_losses = train_model(data_loader,epochs,model,criterion,optimizer,scheduler)
# #         np.save('train_losses of folder {} and fraction {}'.format(j, fraction),train_losses)
# #         np.save('val_losees of folder {} and fraction {}'.format(j, fraction), val_losses)
#         print('Saving after {} fold..'.format(j))
#         state = {
#                     'net': model_output, #.module if use_cuda else net,
#                     'train_loss':train_losses,
#                     'val_loss':val_losses
#                         }
#         if not os.path.isdir('folder_checkpoint_4'):
#             os.mkdir('folder_checkpoint_4')
#         torch.save(state, './folder_checkpoint_4/folder_IN_.t7')


# In[ ]:

# def load_data_kfold(k,sorted_pt_constituentsnp,y_train):
        
#     folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(sorted_pt_constituentsnp,y_train))
    
#     return folds


# In[ ]:

# k = 3
# folds = load_data_kfold(k,sorted_pt_constituentsnp,y_train)


# In[ ]:


#     def create_B(self,x):
#         print("This is x shape {}".format(x.shape))
#         RA = torch.zeros(x.shape[0], 3, self.Nr)
#         RR = torch.zeros(x.shape[0], self.N, self.Nr)
#         RS = torch.zeros(x.shape[0], self.N, self.Nr)
#         for k, arr in enumerate(x):
#             arr = arr.data.cpu().numpy()
#             Rr = torch.zeros(self.N, self.Nr)
#             Rs = torch.zeros(self.N, self.Nr)
#             Ra = torch.zeros(3, self.Nr)
#             for i, (r, s) in enumerate(self.receiver_sender_list):
#                 p = -1
#                 if arr[r, 0]==0 or arr[s,0] == 0:
#                     Rr[r, i] = 0
#                     Rs[s, i] = 0
#                 else:
#                     Rr[r, i] = 1
#                     Rs[s, i] = 1
#                     for j in range(3):
#                         if arr[r, 0]==0 or arr[s,0] == 0:
#                             Ra[j, i] = 0
#                         else:
#                             Ra[j, i] = self.distance(arr[r,5],arr[s,5],arr[r,7],arr[s,7],arr[r,10],arr[s,10],p)
#                         p=p+1
#             print("This is Rr shape {}".format(Rr.shape))
#             print("This is Rs shape {}".format(Rs.shape))
#             print("This is Ra shape {}".format(Ra.shape))
#             RA[k] = Ra
#             RS[k] = Rs
#             RR[k] = Rr
#         print("This is RR shape {}".format(RR.shape))
#         print("This is RS shape {}".format(RS.shape))
#         print("This is RA shape {}".format(RA.shape))
#         RR = Variable(RR).cuda()
#         RS = Variable(RS).cuda()
#         RA = Variable(RA).cuda()
#         return RR,RS,RA


# In[ ]:

# class GraphNet(nn.Module):
#     def __init__(self, n_constituents, n_targets, params, hidden):
#         super(GraphNet, self).__init__()
#         self.hidden = hidden
#         self.P = params
#         self.N = n_constituents
#         self.Nr = self.N * (self.N - 1)
#         self.Dr = 3
#         self.De = 7
#         self.Dx = 0
#         self.Do = 6
#         self.n_targets = n_targets
#         self.fr1 = nn.Linear(2 * self.P + self.Dr, hidden, bias = False)
#         self.fr2 = nn.Linear(hidden, int(hidden/2), bias = False)
#         self.fr3 = nn.Linear(int(hidden/2), self.De, bias = False)
#         self.fo1 = nn.Linear(self.P + self.Dx + self.De, hidden, bias = False)
#         self.fo2 = nn.Linear(hidden, int(hidden/2), bias = False)
#         self.fo3 = nn.Linear(int(hidden/2), self.Do, bias = False)
#         self.fc1 = nn.Linear(self.Do, hidden)
#         self.fc2 = nn.Linear(hidden, int(hidden/2))
#         self.fc3 = nn.Linear(int(hidden/2), self.n_targets)
#         self.receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
#     def distance(self,pt_1,pt_2,eta_1,eta_2,phi_1,phi_2,p):
#         deltaR = (phi_1-phi_2)**2 + (eta_1-eta_2)**2
#         dist = min(pt_1**(2*p), pt_2**(2*p))*deltaR
#         return dist
#     def tmul(self,x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
#         x_shape = x.size()
#         y_shape = y.size()
#         return torch.mm(x, y).view(-1, x_shape[0], y_shape[1])
#     def creation_of_B(self,x):
#         B = torch.zeros(x.shape[0], 2 * self.P + self.Dr, self.Nr).cuda()
#         RR = torch.zeros(x.shape[0], self.N, self.Nr).cuda()
#         for k, arr in enumerate(x):
#             arr = arr.data.cpu().numpy()
#             Rr = torch.zeros(self.N, self.Nr) #188*(188*187)
#             Rs = torch.zeros(self.N, self.Nr)
#             Ra = torch.zeros(3, self.Nr)
#             for i, (r, s) in enumerate(self.receiver_sender_list):
#                 p = -1
#                 if arr[r, 0]==0 or arr[s,0] == 0:
#                     Rr[r, i] = 0
#                     Rs[s, i] = 0
#                 else:
#                     Rr[r, i] = 1
#                     Rs[s, i] = 1
#                     for j in range(3):
#                         if arr[r, 0]==0 or arr[s,0] == 0:
#                             Ra[j, i] = 0
#                         else:
#                             Ra[j, i] = self.distance(arr[r,5],arr[s,5],arr[r,7],arr[s,7],arr[r,10],arr[s,10],p)
#                         p=p+1
#             arr = torch.transpose(torch.from_numpy(arr).type(torch.cuda.FloatTensor), 0, 1).contiguous() # 98710*16*188
#             Rr = Rr.cuda()
#             Rs = Rs.cuda()
#             Ra = Ra.cuda()
#             Orr = self.tmul(arr, Rr)
#             Ors = self.tmul(arr, Rs)
#             Bb = torch.cat([Orr, Ors], 1)
#             Ra=Ra.view(-1,Ra.shape[0],Ra.shape[1])
#             Bb = torch.cat([Bb, Ra], 1) # batch_size* (2P+De)*Nr 
#             Bb= Bb.view(-1, Bb.shape[2]) # cgtob batch_size ubrat'
# #             Bb = torch.transpose(Bb, 1, 2).contiguous()
#             RR[k] = Rr
#             B[k] = Bb
#         del Rr
#         del Bb
#         B=torch.transpose(B, 1, 2).contiguous()
#         return B,RR
#     def multiplication(self,E,RR):
#         Ebar = torch.zeros(E.shape[0], self.De, self.N).cuda()
#         E = E.data
#         RR = RR.data
#         for i, (e, Rr) in enumerate(zip(E,RR)):
#             ebar = torch.mm(e, Rr).view(-1, e.shape[0], Rr.shape[1])
#             Ebar[i] = ebar
#         return Variable(Ebar)
#     def forward(self, x):
#         # x - initial dataset
#         B,RR = self.creation_of_B(x)
#         B= Variable(B).cuda()
#         RR = Variable(RR).cuda()
#         x=torch.transpose(x, 1, 2).contiguous()
#         B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
#         B = nn.functional.relu(self.fr2(B))
#         E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
#         del B
#         E = torch.transpose(E, 1, 2).contiguous()
#         Ebar = self.multiplication(E, torch.transpose(RR, 1, 2))
#         del E
#         C = torch.cat([x, Ebar], 1)
#         del Ebar
#         C = torch.transpose(C, 1, 2).contiguous()
#         ### Second MLP ###
#         C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
#         C = nn.functional.relu(self.fo2(C))
#         O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do)) 
#         O = torch.sum(O,  dim=1)
#         del C
#         ### Classification MLP ###
#         N = nn.functional.relu(self.fc1(O.view(-1, self.Do)))
#         del O
#         N = nn.functional.relu(self.fc2(N))
#         N = self.fc3(N)
#         return N
# model = GraphNet(n_constituents=188,n_targets=5,params=16, hidden=10)
# model = model.cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
# criterion = nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='min',factor=0.5,patience=10)


