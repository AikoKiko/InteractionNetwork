
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, StratifiedKFold


# In[2]:


infile = h5py.File('/bigdata/shared/HLS4ML/NEW/AS/jetImageMerged.h5', 'r')
file_m = h5py.File('/bigdata/shared/HLS4ML/NEW/jetImage_983445_990519.h5', 'r')
jets = infile.get('jets')
jetImageECAL = np.array(infile.get('jetImageECAL'))
jetImageHCAL = np.array(infile.get('jetImageHCAL'))
featureName = np.array(file_m.get("jetFeatureNames"))
print(list(featureName))
target = jets[:,-6:-1]
expFeature_labels = [b'j_zlogz', b'j_c1_b0_mmdt',b'j_c1_b1_mmdt',b'j_c1_b2_mmdt',b'j_c2_b1_mmdt',b'j_c2_b2_mmdt',b'j_d2_b1_mmdt',
                      b'j_d2_b2_mmdt',b'j_d2_a1_b1_mmdt',b'j_d2_a1_b2_mmdt',b'j_m2_b1_mmdt',b'j_m2_b2_mmdt',b'j_n2_b1_mmdt',
                      b'j_n2_b2_mmdt',b'j_mass_mmdt',b'j_multiplicity']
exFeature_indexes = []
for feature  in expFeature_labels :
    featureindex = featureName.tolist().index(feature)
    exFeature_indexes.append(featureindex)
exFeature_indexes.sort() 
print(exFeature_indexes)
expFeature = jets[:,exFeature_indexes]

print(expFeature.shape)


# In[3]:


maxPt = max(jets[:,1])
jetImageECAL = jetImageECAL/maxPt
jetImageHCAL = jetImageHCAL/maxPt


# In[4]:


def shuffle(a,d, b, c):
    iX = a.shape[1]
    iY = a.shape[2]
    b_shape = b.shape[1]
    a = a.reshape(a.shape[0], iX*iY)
    d = d.reshape(d.shape[0], iX*iY)
    total = np.column_stack((a,d))
    total = np.column_stack((total,b))
    total = np.column_stack((total,c))
    np.random.shuffle(total)
    a = total[:,:iX*iY]
    d = total[:,iX*iY:2*iX*iY]
    b = total[:,2*iX*iY:2*iX*iY+b_shape]
    c = total[:,2*iX*iY+b_shape:]
    a = a.reshape(a.shape[0],iX, iY,1)
    d = d.reshape(d.shape[0],iX, iY,1)
    return a,d,b,c,b_shape


# In[5]:


jetImageECAL,jetImageHCAL, expFeature, target, b_shape = shuffle(jetImageECAL,jetImageHCAL, expFeature, target)


# In[6]:


from sklearn.preprocessing import StandardScaler
print(expFeature.shape)
scaler = StandardScaler()
scaled_expFeature = scaler.fit_transform(expFeature)
print(scaled_expFeature.shape)


# In[7]:


y_train_in =np.argmax(target, axis=1)


# In[8]:


epochs=100
batch_size=1024


# In[9]:


iSplit_1 = int(0.6*target.shape[0])
iSplit_2 = int(0.8*target.shape[0])
x_train = scaled_expFeature[:iSplit_1, :]
x_valid = scaled_expFeature[iSplit_1:iSplit_2, :]
x_test = scaled_expFeature[iSplit_2:, :]
y_train= y_train_in[:iSplit_1]
y_valid = y_train_in[iSplit_1:iSplit_2]
y_test = y_train_in[iSplit_2:]
training_dataset = np.column_stack((x_train,y_train))
validation_dataset = np.column_stack((x_valid,y_valid))
testing_dataset = np.column_stack((x_test,y_test))
y_test_label = target[iSplit_2:, :]


# In[16]:


class Net(nn.Module):
    def __init__(self, hidden_layers, dropout, activ_f):
        super(Net, self).__init__()
        self.dropout = dropout
        self.activ_f = activ_f
        self.linear_layers = nn.ModuleList()
        self.input = 16
        for i in range(int(hidden_layers)):
            linear_layer = nn.Linear(self.input,2**(4+i))
            self.linear_layers.append(linear_layer)
            self.input = 2**(4+i)
        self.fc1 = nn.Linear(int(self.input), 5)
    def forward(self, x):
        for i in range(len(self.linear_layers)):
            x= getattr(F, self.activ_f)(self.linear_layers[i](x))
            x = F.dropout(x, p= self.dropout, training=True)
        return F.log_softmax(self.fc1(x))
    
model = Net(5,0,'tanh')
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion= nn.NLLLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='min',factor=0.5,patience=10)

# class Net(nn.Module):
#     def __init__(self, hidden_layers, dropout, activ_f):
#         super(Net, self).__init__()
#         self.dropout = dropout
#         self.activ_f = activ_f
#         hidden =64
#         nodes = [16]
#         for i in range(hidden_layers):
#             if i == hidden_layers-1:
#                 nodes.append(5)
#             else:
#                 nodes.append(hidden)
#                 hidden = int(hidden/2)
#         self.linear_layers = nn.ModuleList()
#         for d in range(1,len(nodes)):
#             linear_layer = torch.nn.Linear(nodes[d-1],nodes[d])
#             self.linear_layers.append(linear_layer)
# #             print(self.linear_layers)
#     def forward(self, x):
#         for i in range (len(self.linear_layers)):
#             if i == len(self.linear_layers)-1:
#                 return F.log_softmax(self.linear_layers[i](x))
#             x = getattr(F, self.activ_f)(self.linear_layers[i](x))
#             x = F.dropout(x, p=self.dropout, training=True)
            
# model = Net()
# model.cuda()
# print(model)
# optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
# criterion= nn.NLLLoss()
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='min',factor=0.5,patience=10)


# In[11]:


class CustomDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:,:b_shape])
        self.y_data = torch.from_numpy(data[:, b_shape:]).squeeze(1)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len


# In[12]:


training_dataset= CustomDataset(training_dataset)
validation_dataset = CustomDataset(validation_dataset)
testing_dataset = CustomDataset(testing_dataset)
train_loader = torch.utils.data.DataLoader(dataset = training_dataset, batch_size = batch_size, shuffle =True, num_workers = 4)
valid_loader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle =True, num_workers = 4)
test_loader = torch.utils.data.DataLoader(dataset = testing_dataset , batch_size = batch_size, shuffle =False, num_workers = 4)
data_loader = {"training" :train_loader, "validation" : valid_loader} 


# In[13]:


class EarlyStopping():
    """
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self, 
                 monitor='val_loss',
                 min_delta=0,
                 patience=10):
        super(EarlyStopping, self).__init__()
        
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e-15
        self.stopped_epoch = 0
        self.stop_training= False
        print("This is my patience {}".format(patience))
    def on_train_begin(self):
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


# In[14]:


def train_model(num_epochs, model, criterion, optimizer,scheduler,volatile=False):

    best_model = model.state_dict()
    best_acc = 0.0
    train_losses ,val_losses = [],[]
    Early_Stopping = EarlyStopping(patience=20)
    Early_Stopping.on_train_begin()
    breakdown = False
    for epoch in range(num_epochs):
        if breakdown:
            print("Early Stopped")
            break
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                if not os.path.isdir('checkpoint_DNN'):
                    os.mkdir('checkpoint_DNN')
                torch.save(state, './checkpoint_DNN/DNN.t7')
                best_acc = epoch_acc
                best_model = model.state_dict()
            if phase == 'validation':
                breakdown = Early_Stopping.on_epoch_end(epoch,round(epoch_loss, 5))
                
            print()

    
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model)
    return model,train_losses ,val_losses


# In[17]:


model_output,train_losses_dense,val_losses_dense = train_model(epochs,model,criterion,optimizer,scheduler)


# In[18]:


plt.figure()
plt.plot(train_losses_dense)
plt.plot(val_losses_dense)
plt.plot()
plt.yscale('log')
plt.title('dense')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[ ]:


print()


# In[19]:


# run a test loop
predicted = np.zeros((x_test.shape[0], 5))
test_loss = 0
test_correct = 0
for batch_idx, (x_data, y_data) in enumerate (test_loader):
    x_data, y_data = Variable(x_data.cuda().type(torch.cuda.FloatTensor),volatile=True),Variable(y_data.cuda().type(torch.cuda.LongTensor))
    model_out = model_output(x_data)
    beg = batch_size*batch_idx
    end = min((batch_idx+1)*batch_size, x_test.shape[0])
    predicted[beg:end] = model_out.data.cpu().numpy()
        
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


# In[ ]:


# x_test_tens=Variable(torch.from_numpy(x_test), volatile=True).float().cuda()
# predict_test = model_output(x_test_tens)


# In[ ]:


# print(predict_test.shape)


# In[ ]:


# print(predict_test)


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
predict_test = np.exp(predicted)
labels_val = y_test_label
df = pd.DataFrame()
fpr = {}
tpr = {}
auc1 = {}

plt.figure()
for i, label in enumerate(labels):
        df[label] = labels_val[:,i]
        df[label + '_pred_dense'] = predict_test[:,i]

        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred_dense'])

        auc1[label] = auc(fpr[label], tpr[label])

        plt.plot(tpr[label],fpr[label],label='%s tagger, auc = %.1f%%'%(label,auc1[label]*100.))
df.to_csv("Dense.csv", sep='\t')
plt.semilogy()
plt.xlabel("sig. efficiency")
plt.ylabel("bkg. mistag rate")
plt.ylim(0.0001,1)
plt.grid(True)
plt.legend(loc='upper left')
#plt.savefig('%s/ROC.pdf'%(options.outputDir))
plt.show()

