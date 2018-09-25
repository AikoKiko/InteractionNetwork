
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
epochs = 100
infile=h5py.File("/bigdata/shared/HLS4ML/NEW/AS/jetImageMerged.h5","r")
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


# In[6]:

test_size = 0.2
valid_size = 0.25
num_train = sorted_pt_constituentsnp.shape[0]
split = int(np.floor(test_size * num_train))
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


# In[7]:

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


# In[9]:

class SimpleNet(nn.Module):
    def __init__(self, gru_layers, dense_layers, hidden,layers,dropout_g,dropout_d,actif_func):
        super(SimpleNet, self).__init__()
        self.layers = nn.ModuleList()
        self.input =16
        self.actif_func = actif_func
        self.dropout_g = dropout_g
        self.dropout_d = dropout_d
        self.gru_layers = int(gru_layers)
        self.dense_layers = int(dense_layers)
        self.hidden = int(hidden)
        self.output = 100
        for i in range(self.gru_layers):
                gru_layer = nn.GRU(input_size=self.input,hidden_size=self.hidden,num_layers=int(layers),batch_first=True,dropout=self.dropout_g)
                self.layers.append(gru_layer)
                self.input = self.hidden
                self.hidden = self.hidden-50
        self.node = self.hidden+50
        self.avg = nn.AdaptiveAvgPool1d(self.node)
        for i in range(self.dense_layers):
                linear_layer = nn.Linear(self.node,self.output)
                self.layers.append(linear_layer)
                self.node = self.output
                self.output = self.output-20
        self.fc1 = nn.Linear(self.output+20, 5)
    def forward(self,x):
        for i in range (self.gru_layers):
            x,_ = self.layers[i](x)
        x, _ = pad_packed_sequence(x, batch_first=True)    
        batch_size,max_len,features = x.size()
        x = x.contiguous().view(batch_size, -1).unsqueeze(0)
        x = self.avg(x).squeeze(0) 
        for i in range(self.dense_layers):
            i = i+self.gru_layers
            x= getattr(F, self.actif_func)(self.layers[i](x))
            x = F.dropout(x, p= self.dropout_d, training=True)
        return F.log_softmax(self.fc1(x))


# In[9]:

class EarlyStopping(object):
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
    Early_Stopping = EarlyStopping(patience=5)
    Early_Stopping.on_train_begin()
    breakdown = False
    for epoch in range(num_epochs):
        if breakdown:
            print("Early Stopped")
            break
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('_' * 10)
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
    model.load_state_dict(best_model)
    return best_acc


# In[ ]:

import os
from threading import Thread
import hashlib
import json
import time
import glob


# In[ ]:

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


# In[ ]:

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


# In[ ]:

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


# In[ ]:

def dummy_func( X ):
    model = SimpleNet(X[0],X[1],X[2],X[3],X[4],X[5],X[6])
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),X[7])
    criterion = nn.NLLLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='min',factor=0.5,patience=10)
    best_acc = train_model(epochs,model,criterion,optimizer,scheduler)
    Y = - (best_acc)
    return Y


# In[ ]:

if __name__ == "__main__":
    from skopt import Optimizer
    from skopt.learning import GaussianProcessRegressor
    from skopt.space import Real, Categorical, Integer
    from skopt import gp_minimize

    import sys
    
    n_par = 8

    externalize = externalfunc(prog='python run_train_exm.py',
                               names = ['par%s'%d for d in range(n_par)]) # open the json file, and write the results 
                                   # for each parameter combination  (just initialization)
    
    run_for = 20
    
    use_func = externalize
    #gru_layers, dense_layers, hidden,layers,dropout_g,dropout_d,actif_func
    dim = [Integer(1, 3),Integer(1, 3),Integer(200,400),Integer(1, 3),Real(0, 0.9),Real(0, 0.9),Categorical(['relu', 'tanh','selu','leaky_relu']),Real(1e-5,1e-3)] # eto i X
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

