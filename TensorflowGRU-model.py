
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import h5py


# In[2]:


num_epochs = 200
batch_size = 64
learning_rate = 0.001
num_input = 16 # MNIST data input (img shape: 28*28)
timesteps = 188 # timesteps
num_hidden = 300
num_classes = 5
display_step = 200 # hz, zachem 


# In[3]:


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


# In[6]:


X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])


# In[7]:


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

# training_data = sorted_pt_constituentsnp
# y_train = mass_targets


# In[8]:


# weights = {
#     'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
# }
# biases = {
#     'out': tf.Variable(tf.random_normal([num_classes]))
# }


# In[9]:


def GRU(x):
    x = tf.unstack(x, timesteps, 1)
    GRU_cell = rnn.GRUCell(num_hidden)
    outputs, states = rnn.static_rnn(GRU_cell, x, dtype=tf.float32)
    print(states.shape)
    dense1 = tf.layers.dense(inputs=outputs[-1], units=100, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense1, units=5)
    return logits


# In[10]:


logits = GRU(X)
# look that when testing it will use softmax directly 
prediction = tf.nn.softmax(logits)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#for testing Aidan
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[11]:


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# In[12]:


# Start training
with tf.Session() as sess:
   # Run the initializer
    sess.run(init)
    for epoch in range(num_epochs):
            running_loss = 0.0
            running_corrects = 0
            for batch_idx in range(int(training_data.shape[0]/batch_size)+1):
                beg = batch_size*batch_idx
                end = min((batch_idx+1)*batch_size, training_data.shape[0])
                batch_x, batch_y = training_data[beg:end], y_train[beg:end]
#                 batch_x = batch_x.reshape((batch_size, timesteps, num_input))
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})     
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                running_loss +=loss
                running_corrects +=acc
            
            epoch_loss = running_loss / training_data.shape[0]
            print('Loss: {:.4f}'.format(
                 epoch_loss))

