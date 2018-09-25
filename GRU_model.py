
# coding: utf-8

# In[43]:


import os
import setGPU
import numpy as np
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
from sklearn.preprocessing import OneHotEncoder


# In[2]:


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


# In[3]:


batch_size = 64
epochs = 200
infile=h5py.File("/bigdata/shared/HLS4ML/jetImage.h5","r")
sorted_pt_constituents = np.array(infile.get('jetConstituentList'))
scaled_jets = infile.get('jets') # in any case
mass_targets = scaled_jets[:,-6:-1]


# In[4]:


def shuffle(a,b):
    iX = a.shape[1]
    iY = a.shape[2]
    b_shape = b.shape[1]
    a = a.reshape(a.shape[0], iX*iY)
    total = np.column_stack((a,b))
    np.random.shuffle(total)
    a = total[:,:iX*iY]
    b = total[:,iX*iY:iX*iY+b_shape]
    a = a.reshape(a.shape[0],iX, iY)
    return a,b


# In[5]:


sorted_pt_constituentsnp, mass_targets = shuffle(sorted_pt_constituents, mass_targets)


# In[18]:


test_size = 0.2
valid_size = 0.25
num_train = sorted_pt_constituentsnp.shape[0]
split = int(np.floor(test_size * num_train))
#for commenting
sorted_indices = list(range(num_train))
split_v=int(np.floor(valid_size * (num_train-split)))
train_idx, test_idx = sorted_indices[split:], sorted_indices[:split]
train_idx, valid_idx = train_idx[split_v:], train_idx[:split_v]

training_data = sorted_pt_constituentsnp[train_idx, :]
validation_data = sorted_pt_constituentsnp[valid_idx, :]
testing_data = sorted_pt_constituentsnp[test_idx,:]
y_train = mass_targets[train_idx, :]
y_valid = mass_targets[valid_idx, :]
y_test = mass_targets[test_idx, :]


# In[19]:


y_train_1D =np.argmax(y_train, axis=1)
y_valid_1D = np.argmax(y_valid, axis=1)
y_test_1D = np.argmax(y_test, axis=1)
train = EventDataset(training_data,y_train_1D)  
valid = EventDataset(validation_data,y_valid_1D) 
test = EventDataset(testing_data,y_test_1D) 
train_loader = DataLoader(dataset = train, batch_size = batch_size, shuffle = True, num_workers = 4)
valid_loader = DataLoader(dataset = valid, batch_size = batch_size, shuffle = True, num_workers = 4) 
test_loader = DataLoader(dataset = test, batch_size = batch_size, shuffle = False, num_workers = 4) 
data_loader = {"training" :train_loader, "validation" : valid_loader} 


# In[8]:


import torch.nn.functional as F
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.gru = nn.GRU(input_size=16, hidden_size=300, num_layers=2, batch_first=True, dropout=0.5)
        
        self.linear1 = nn.Linear(300,100)
        self.linear2 = nn.Linear(100,5)
        self.avg = nn.AdaptiveAvgPool1d(300)
    def forward(self, constituents):
        #self.gru.flatten_parameters()
        con_new, con_prev = self.gru(constituents)
        #print("con_new[1] = {}".format(con_new[1]))
        #con = con_new[0]
        #print("con_new shape = {}".format(con_new[0].shape))
        #con = con.view(batch_size,-1).unsqueeze(dim=0)
        #print("reshaped shape = {}".format(con.shape))
        #con = self.avg(con).squeeze()
        #print("avg pool shape = {}".format(con))
        
        con, _ = pad_packed_sequence(con_new, batch_first=True)
        #print("pad_packed output: {}".format(con.shape))
        batch_size,max_len, features = con.size()
        con = con.contiguous().view(batch_size, -1).unsqueeze(0)
        #print("reshaped output = {}".format(con.shape))
        con = self.avg(con).squeeze(0)
        #print("pooled output = {}".format(con.shape))
        #print("reshape output: {}".format(con.shape))
        #con = con[1] # need to clarify
#         con = F.dropout(con, p=0.25, training=True)
        con =F.relu(self.linear1(con))
#         con = F.dropout(con, p=0.25, training=True)
        #con = F.dropout(con, p=0.25, training=True)
        con = self.linear2(con)
        #con = con.view(batch_size, -1)
        return con 
model = SimpleNet()
model.cuda()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='min',factor=0.5,patience=10)


# In[9]:


class EarlyStopping():
    """
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self, 
                 monitor='val_loss',
                 min_delta=0,
                 patience=25):
        super(EarlyStopping, self).__init__()
        
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
        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
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


# In[10]:


def train_model(num_epochs, model, criterion, optimizer,scheduler,volatile=False):

    best_model = model.state_dict()
    best_acc = 0.0
    train_losses ,val_losses = [],[]
    Early_Stopping = EarlyStopping(patience=15)
    Early_Stopping.on_train_begin()
    breakdown = False
    for epoch in range(num_epochs):
        if breakdown:
            print("Early Stopped")
#             f.write('Early Stopped\n') 
            break
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         f.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        print('_' * 10)
#         f.write('_\n' * 10)
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
                x_data=x_data.cpu().numpy()
                y_data=y_data.cpu().numpy()
                arr = np.sum(x_data!=0, axis=1)[:,0]
                arr=[1 if x==0 else x for x in arr]
                arr = np.array(arr)
                sorted_indices_la= np.argsort(-arr)
                x_data = x_data[sorted_indices_la, ...]
                y_data = y_data[sorted_indices_la]
                x_data = torch.from_numpy(x_data)
                y_data = torch.from_numpy(y_data)
                t_seq_length= [ arr[i] for i in sorted_indices_la]
                x_data, y_data = Variable(x_data.cuda().type(torch.cuda.FloatTensor),volatile),Variable(y_data.cuda().type(torch.cuda.LongTensor))
                x_data = torch.nn.utils.rnn.pack_padded_sequence(x_data, t_seq_length, batch_first=True)
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
#             f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(
#                 phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
#                 print('Saving..')
#                 state = {
#                         'net': model, #.module if use_cuda else net,
#                         'epoch': epoch,
#                         'best_acc':epoch_acc,
#                         'train_loss':train_losses,
#                         'val_loss':val_losses,
#                         }
#                 if not os.path.isdir('checkpoint_Thong'):
#                     os.mkdir('checkpoint_Thong')
#                 torch.save(state, './checkpoint_Thong/IN.t7')
                best_acc = epoch_acc
                best_model = model.state_dict()
            if phase == 'validation':
                breakdown = Early_Stopping.on_epoch_end(epoch,round(epoch_loss,4))
                

    
    print('Best val Acc: {:4f}'.format(best_acc))
#     f.write('Best val Acc: {:4f}\n'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model)
    return model,train_losses ,val_losses


# In[11]:


model_output,train_losses,val_losses = train_model(epochs,model,criterion,optimizer,scheduler)


# In[12]:


# model = model_output
# name = "GRU_des_pt_avg_pooling"
# with open("nfshome/aidanaserikova/jetTagge_GRU_avg_pool_%s.pt" %name, "w") as file:
#     torch.save(model, file)


# In[34]:


# testing_data_tens=Variable(torch.from_numpy(testing_data),volatile=True).float().cuda()
#packed_cons_test = torch.nn.utils.rnn.pack_padded_sequence(testing_data_tens, ts_seq_length, batch_first=True)
# labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
# labels_val = y_test 


# In[14]:


#predict_test= predict_test.data.cpu().numpy()


# In[15]:


# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt
# df = pd.DataFrame()
# fpr = {}
# tpr = {}
# auc1 = {}
# predict_test = model_output(testing_data_tens)
# predict_test= predict_test.data.cpu().numpy()
# plt.figure()
# for i, label in enumerate(labels):
#         df[label] = labels_val[:,i]
#         df[label + '_pred'] = predict_test[:,i]

#         fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred'])

#         auc1[label] = auc(fpr[label], tpr[label])

#         plt.plot(tpr[label],fpr[label],label='%s tagger, auc = %.1f%%'%(label,auc1[label]*100.))
# plt.semilogy()
# plt.xlabel("sig. efficiency")
# plt.ylabel("bkg. mistag rate")
# plt.ylim(0.0001,1)
# plt.grid(True)
# plt.legend(loc='upper left')
# #plt.savefig('%s/ROC.pdf'%(options.outputDir))
# plt.show()


# In[41]:


predicted = np.zeros((19742, 5))
truth = np.zeros((19742,))
test_loss = 0
test_correct = 0
for batch_idx, (x_data, y_data) in enumerate (test_loader):
    x_data=x_data.cpu().numpy()
    y_data=y_data.cpu().numpy()
    arr = np.sum(x_data!=0, axis=1)[:,0]
    arr=[1 if x==0 else x for x in arr]
    arr = np.array(arr)
    sorted_indices_la= np.argsort(-arr)
    x_data = x_data[sorted_indices_la, ...]
    y_data = y_data[sorted_indices_la]
    x_data = torch.from_numpy(x_data)
    y_data = torch.from_numpy(y_data)
    t_seq_length= [ arr[i] for i in sorted_indices_la]
    x_data, y_data = Variable(x_data.cuda().type(torch.cuda.FloatTensor),volatile=True),Variable(y_data.cuda().type(torch.cuda.LongTensor))
    x_data = torch.nn.utils.rnn.pack_padded_sequence(x_data, t_seq_length, batch_first=True)
    model_out = model_output(x_data)
    #print(model_output.data.shape)
    beg = batch_size*batch_idx
    end = min((batch_idx+1)*batch_size, 19742)
    predicted[beg:end] = F.softmax(model_out).data.cpu().numpy()
    truth[beg:end] = y_data.data.cpu().numpy()
        
    # sum up batch loss
    test_loss += criterion(model_out, y_data).data[0]
    _, preds = torch.max(model_out.data, 1)  # get the index of the max log-probability
    #predictions.extend(preds)
    test_correct += preds.eq(y_data.data).sum()

test_loss /= len(test_loader.dataset)
test_acc = 100. * test_correct/ len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, len(test_loader.dataset),
        test_acc))
# f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, test_correct, len(test_loader.dataset),
#         test_acc))


# In[37]:


arr = np.sum(testing_data!=0, axis=1)[:,0]
arr=[1 if x==0 else x for x in arr]
arr = np.array(arr)
sorted_indices_la= np.argsort(-arr) #NOOO. because you are doing global sort. vs batch sort in x. but dayaset was not shuffled 
# Global: 3 4 6 7 8 2 0
# batch size = 2
# batch 1: 3 4
# b2: 6 7 
# b3: 2 8
# b4: 0
# global sort: 0 2 3 4 6 7 8
# x: 3 4 6 7 2 8 0
y_test = y_test[sorted_indices_la]


# In[48]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
predict_test = predicted
enc = OneHotEncoder(5, sparse = False)
labels_val = enc.fit_transform(truth.reshape((19742,1)))
df = pd.DataFrame()
fpr = {}
tpr = {}
auc1 = {}

plt.figure()
for i, label in enumerate(labels):
        df[label] = labels_val[:,i]
        df[label + '_pred_GRU'] = predict_test[:,i]

        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred_GRU'])

        auc1[label] = auc(fpr[label], tpr[label])

        plt.plot(tpr[label],fpr[label],label='%s tagger, auc = %.1f%%'%(label,auc1[label]*100.))
df.to_csv("GRU.csv", sep='\t')
plt.semilogy()
plt.xlabel("sig. efficiency")
plt.ylabel("bkg. mistag rate")
plt.ylim(0.0001,1)
plt.grid(True)
plt.legend(loc='upper left')
#plt.savefig('%s/ROC.pdf'%(options.outputDir))
plt.show()

