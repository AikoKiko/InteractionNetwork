
# coding: utf-8

# In[1]:


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
import itertools


# In[2]:


N=188
infile = h5py.File('/bigdata/shared/HLS4ML/jetMerged', 'r')
sorted_pt_constituents = np.array(infile.get('jetConstituentList'))
scaled_jets = infile.get('jets') # in any case
mass_targets = scaled_jets[:,-6:-1]
print("Loading completed")


# In[3]:


import torch.nn.functional as F
class GraphNet(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden):
        super(GraphNet, self).__init__()
        self.hidden = hidden
        self.P = params
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Dr = 0
        self.De = 5
        self.Dx = 0
        self.Do = 6
        self.n_targets = n_targets
        self.assign_matrices()
        self.Ra = Variable(torch.ones(self.Dr, self.Nr))
        self.fr1 = nn.Linear(2 * self.P + self.Dr, hidden)
        self.fr2 = nn.Linear(hidden, int(hidden/2))
        self.fr3 = nn.Linear(int(hidden/2), self.De)
        self.fo1 = nn.Linear(self.P + self.Dx + self.De, hidden)
        self.fo2 = nn.Linear(hidden, int(hidden/2))
        self.fo3 = nn.Linear(int(hidden/2), self.Do)
        self.fc1 = nn.Linear(self.Do * self.N, hidden)
        self.fc2 = nn.Linear(hidden, int(hidden/2))
        self.fc3 = nn.Linear(int(hidden/2), self.n_targets)
    
    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = Variable(self.Rr).cuda()
        self.Rs = Variable(self.Rs).cuda()
        
    def forward(self, x):
        x=torch.transpose(x, 1, 2).contiguous()
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
        B = nn.functional.relu(self.fr2(B))
        E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
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
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])

    
model = GraphNet(n_constituents=188,n_targets=5,params=16, hidden=10)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='min',factor=0.5,patience=10)


# In[4]:


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


# In[5]:


test_size = 0.2
valid_size = 0.25
batch_size = 20
pin_memory = False
num_workers = 4
epochs = 200


# In[6]:


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


# In[7]:


sorted_pt_constituents, mass_targets = shuffle(sorted_pt_constituents, mass_targets)


# In[8]:


#for commenting
sorted_pt_constituentsnp= sorted_pt_constituents

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



# In[9]:


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
####### finished 


# In[10]:


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


# In[11]:


f= open('INresults-JR_suggest.txt', 'w')


# In[14]:


def train_model(num_epochs, model, criterion, optimizer,scheduler,volatile=False):

    best_model = model.state_dict()
    best_acc = 0.0
    train_losses ,val_losses = [],[]
    Early_Stopping = EarlyStopping(patience=30)
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
                x_data, y_data = Variable(x_data.cuda().type(torch.cuda.FloatTensor),volatile),Variable(y_data.cuda().type(torch.cuda.LongTensor))
                print("This is {} phase, and data size is {}" .format(phase, x_data.shape))
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
                print('Saving..')
                state = {
                        'net': model, #.module if use_cuda else net,
                        'epoch': epoch,
                        'best_acc':epoch_acc,
                        'train_loss':train_losses,
                        'val_loss':val_losses,
                        }
                if not os.path.isdir('checkpoint_JR'):
                    os.mkdir('checkpoint_JR')
                torch.save(state, './checkpoint_JR/IN.t7')
                best_acc = epoch_acc
                best_model = model.state_dict()
            if phase == 'validation':
                breakdown = Early_Stopping.on_epoch_end(epoch,round(epoch_loss,4))
                

    
    print('Best val Acc: {:4f}'.format(best_acc))
#     f.write('Best val Acc: {:4f}\n'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model)
    return model,train_losses ,val_losses


# In[16]:


model_output,train_losses,val_losses = train_model(epochs,model,criterion,optimizer,scheduler)


# In[ ]:


# train_losses =np.array(train_losses)
# val_losses = np.array(val_losses)
# np.save('JRtrain_losses',train_losses)
# np.save('JRval_losees', val_losses)


# In[ ]:


predicted = np.zeros((19742, 5))
test_loss = 0
test_correct = 0
for batch_idx, (x_data, y_data) in enumerate (test_loader):
    x_data, y_data = Variable(x_data.cuda().type(torch.cuda.FloatTensor),volatile=True),Variable(y_data.cuda().type(torch.cuda.LongTensor))
    model_out = model_output(x_data)
    #print(model_output.data.shape)
    beg = batch_size*batch_idx
    end = min((batch_idx+1)*batch_size, 19742)
    predicted[beg:end] = F.softmax(model_out).data.cpu().numpy()
        
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


# In[ ]:


# np.save('predictedJR',predicted)
# f.close()


# In[ ]:


# print(predicted)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
predict_test = np.exp(predicted)
labels_val = y_test
df = pd.DataFrame()
fpr = {}
tpr = {}
auc1 = {}

plt.figure()
for i, label in enumerate(labels):
        df[label] = labels_val[:,i]
        df[label + '_pred_IN'] = predict_test[:,i]

        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred_IN'])

        auc1[label] = auc(fpr[label], tpr[label])

        plt.plot(tpr[label],fpr[label],label='%s tagger, auc = %.1f%%'%(label,auc1[label]*100.))
df.to_csv("IN.csv", sep='\t')
plt.semilogy()
plt.xlabel("sig. efficiency")
plt.ylabel("bkg. mistag rate")
plt.ylim(0.0001,1)
plt.grid(True)
plt.legend(loc='upper left')
#plt.savefig('%s/ROC.pdf'%(options.outputDir))
plt.show()


# In[ ]:


plt.plot(train_losses)
plt.plot(val_losses)
plt.plot()
plt.yscale('log')
plt.title('IN')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

