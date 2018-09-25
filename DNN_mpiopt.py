
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
from sklearn.model_selection import train_test_split, StratifiedKFold


# In[3]:


infile = h5py.File('/bigdata/shared/HLS4ML/NEW/AS/jetImageMerged.h5', 'r')
file_d = h5py.File('/bigdata/shared/HLS4ML/NEW/jetImage_3507666_3511615.h5', 'r')
jets = infile.get('jets')
jetImageECAL = np.array(infile.get('jetImageECAL'))
jetImageHCAL = np.array(infile.get('jetImageHCAL'))
featureName = np.array(file_d.get("jetFeatureNames"))
target = jets[:,-6:-1]
expFeature_labels = [b'j_zlogz', b'j_c1_b0_mmdt',b'j_c1_b1_mmdt',b'j_c1_b2_mmdt',b'j_c2_b1_mmdt',b'j_c2_b2_mmdt',b'j_d2_b1_mmdt',
                      b'j_d2_b2_mmdt',b'j_d2_a1_b1_mmdt',b'j_d2_a1_b2_mmdt',b'j_m2_b1_mmdt',b'j_m2_b2_mmdt',b'j_n2_b1_mmdt',
                      b'j_n2_b2_mmdt',b'j_mass_mmdt',b'j_multiplicity']
exFeature_indexes = []
for feature  in expFeature_labels :
    featureindex = featureName.tolist().index(feature)
    exFeature_indexes.append(featureindex)
exFeature_indexes.sort() 
expFeature = jets[:,exFeature_indexes]


# In[4]:


maxPt = max(jets[:,1])
jetImageECAL = jetImageECAL/maxPt
jetImageHCAL = jetImageHCAL/maxPt


# In[5]:


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


# In[6]:


jetImageECAL,jetImageHCAL, expFeature, target, b_shape = shuffle(jetImageECAL,jetImageHCAL, expFeature, target)


# In[7]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_expFeature = scaler.fit_transform(expFeature)


# In[8]:


y_train_in =np.argmax(target, axis=1)
epochs=200
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


# In[10]:


class CustomDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:,:b_shape])
        self.y_data = torch.from_numpy(data[:, b_shape:]).squeeze(1)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len


# In[11]:


training_dataset= CustomDataset(training_dataset)
validation_dataset = CustomDataset(validation_dataset)
testing_dataset = CustomDataset(testing_dataset)
train_loader = torch.utils.data.DataLoader(dataset = training_dataset, batch_size = batch_size, shuffle =True, num_workers = 4)
valid_loader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle =True, num_workers = 4)
test_loader = torch.utils.data.DataLoader(dataset = testing_dataset , batch_size = batch_size, shuffle =False, num_workers = 4)
data_loader = {"training" :train_loader, "validation" : valid_loader} 


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


# In[14]:


class EarlyStopping(object):
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


# In[15]:


def train_model(num_epochs, model, criterion, optimizer,scheduler,volatile=False):
    best_model = model.state_dict()
    best_acc = 0.0
    train_losses ,val_losses = [],[]
    Early_Stopping = EarlyStopping(patience=30)
    Early_Stopping.on_train_begin()
    breakdown = False
    p=10
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
                if not os.path.isdir('checkpoint4'):
                    os.mkdir('checkpoint4')
                torch.save(state, './checkpoint4/IN.t7')
                best_acc = epoch_acc
                best_model = model.state_dict()
            if phase == 'validation':
                breakdown = Early_Stopping.on_epoch_end(epoch,round(epoch_loss, 4))
            print() 
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model)
    return best_acc


# In[35]:


import os
from threading import Thread
import hashlib
import json
import time
import glob


# In[36]:


class externalfunc: # call the python_ex_blablabla.py function, get the accuracy , write to the json file and then read it 
                    #and return accuracy
    def __init__(self , prog, names):
        self.call = prog
        self.N = names
        
    def __call__(self, X):  # eto i est' chto tam gp.mimimize call the function with dim 
        self.args = dict(zip(self.N,X))
        h = hashlib.md5(str(self.args).encode('utf-8')).hexdigest()
        com = '%s %s'% (self.call, ' '.join(['--%s %s'%(k,v) for (k,v) in self.args.items() ])) # koroch delaet python lll.py
                           # par0 0.45, par1 0.75 ....
        com += ' --hash %s'%h
        com += ' > %s.log'%h
        print ("Executing: ",com)
        ## run the command
        c = os.system( com )
        ## get the output
        try:
            r = json.loads(open('%s.json'%h).read())
            Y = r['result']
        except:
            print ("Failed on",com)
            Y = None
        return Y   # vernet accuracy po tem parametram chto on tol'ko chto zatestil 


# In[37]:


class manager:
    def __init__(self, n, skobj,
                 iterations, func, wait=10):
        self.n = n ## number of parallel processes, smth related with workers
        self.sk = skobj ## the skoptimizer you created
        self.iterations = iterations # run_for
        self.wait = wait
        self.func = func   
    def run(self):

        ## first collect all possible existing results
        for eh  in  glob.glob('*.json'):
            try:
                ehf = json.loads(open(eh).read())
                y = ehf['result']
                x = [ehf['params'][n] for n in self.func.N] # par0 , par1, par2
                print ("pre-fitting",x,y,"remove",eh,"to prevent this")
                print (skop.__version__)
                self.sk.tell( x,y )
            except:
                pass
        workers=[]
        it = 0
        asked = []
        while it< self.iterations:
            ## number of thread going
            n_on = sum([w.is_alive() for w in workers])
            if n_on< self.n:
                ## find all workers that were not used yet, and tell their value
                XYs = []
                for w in workers:
                    if (not w.used and not w.is_alive()):
                        if w.Y != None:
                            XYs.append((w.X,w.Y))
                        w.used = True
                    
                if XYs:
                    one_by_one= False
                    if one_by_one:
                        for xy in XYs:
                            print ("\t got",xy[1],"at",xy[0])
                            self.sk.tell(xy[0], xy[1])
                    else:
                        print ("\t got",len(XYs),"values")
                        print ("\n".join(str(xy) for xy in XYs))
                        self.sk.tell( [xy[0] for xy in XYs], [xy[1] for xy in XYs])
                    asked = [] ## there will be new suggested values
                    print (len(self.sk.Xi))

                        
                ## spawn a new one, with proposed parameters
                if not asked:
                    asked = self.sk.ask(n_points = self.n)
                if asked:
                    par = asked.pop(-1)
                else:
                    print ("no value recommended")
                it+=1
                print ("Starting a thread with",par,"%d/%d"%(it,self.iterations))
                workers.append( worker(
                    X=par ,
                    func=self.func ))
                workers[-1].start()
                time.sleep(self.wait) ## do not start all at the same exact time
            else:
                ## threads are still running
                if self.wait:
                    #print n_on,"still running"
                    pass
                time.sleep(self.wait)


# In[38]:


class worker(Thread):
    def __init__(self,
                 #N,
                 X,
                 func):
        Thread.__init__(self)
        self.X = X
        self.used = False
        self.func = func
        
    def run(self):
        self.Y = self.func(self.X)


# In[39]:


def dummy_func( X ):
    model = Net(X[0],X[1],X[2])
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=X[3])
    criterion= nn.NLLLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='min',factor=0.5,patience=10)
    best_acc = train_model(epochs,model,criterion,optimizer,scheduler)
    Y = - (best_acc)
    return Y


# In[40]:


if __name__ == "__main__":
    from skopt import Optimizer
    from skopt.learning import GaussianProcessRegressor
    from skopt.space import Real, Categorical, Integer
    from skopt import gp_minimize

    import sys
    
    n_par = 4

    externalize = externalfunc(prog='python run_train_ex.py',
                               names = ['par%s'%d for d in range(n_par)]) # open the json file, and write the results 
                                   # for each parameter combination  (just initialization)
    
    run_for = 20

    use_func = externalize
    dim = [Integer(1, 5), Real(0, 0.9),Categorical(['relu', 'tanh','selu','leaky_relu']),Real(1e-5,1e-3)] # eto i X
    start = time.mktime(time.gmtime())
    res = gp_minimize(
        func=use_func,
        dimensions=dim,
        n_calls = run_for,
        )

    print ("GPM best value",res.fun,"at",res.x) # function value at the minimum and location of the minimum 
    #print res
    print ("took",time.mktime(time.gmtime())-start,"[s]")
    
    
    o = Optimizer(
        n_initial_points =5,
        acq_func = 'gp_hedge',
        acq_optimizer='auto',
        base_estimator=GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                                                n_restarts_optimizer=2,
                                                noise='gaussian', normalize_y=True,
                                                optimizer='fmin_l_bfgs_b'),
        dimensions=dim,
    )

    m = manager(n = 4,
                skobj = o,
                iterations = run_for,
                func = use_func,
                wait= 0
    )
    start = time.mktime(time.gmtime())
    m.run() # nado posmotret' chto manager .run delaet
    import numpy as np
    best = np.argmin( m.sk.yi)
    print ("Threaded GPM best value",m.sk.yi[best],"at",m.sk.Xi[best],)
    print ("took",time.mktime(time.gmtime())-start,"[s]")

