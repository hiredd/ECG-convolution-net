# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:25:27 2017

@author: parker
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import wfdb
import csv
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import tensorflow as tf

from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Conv1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt


def process_qrs_index(file_name_lst):
    
    lst_data = []
    label = []
    
    for var_file in file_name_lst:
        anotation = wfdb.rdann(var_file, 'atr')
        #label        
        ty = anotation.anntype
        #segment index        
        index = anotation.annsamp
        record = wfdb.rdsamp(var_file, physical=False, channels = [0])
        d_sig = record.d_signals
        for i in range(1,len(index) - 1):
            start = index[i]
            end = index[i+1]
            temp_lst = d_sig[start : end, 0].tolist()
            if ty[i+1] == 'N':
                label.append(0)
            if ty[i+1] == 'A':
                label.append(1)
            if ty[i+1] == 'E':
                label.append(2)
            if ty[i+1] == 'F':
                label.append(3)
            if ty[i+1] == 'L':
                label.append(4)
            if ty[i+1] == 'R':
                label.append(5)
            if ty[i+1] == 'V':
                label.append(6)
            if ty[i+1] == 'a':
                label.append(7)
            if ty[i+1] == 'f':
                label.append(8)
            if ty[i+1] == 'j':
                label.append(9)
            if ty[i+1] == '/':
                label.append(10)
            #temp_lst.append(ty[i+1])
            if ty[i+1] == '/' or ty[i+1] == 'A' or ty[i+1] == 'E' or ty[i+1] == 'F' or ty[i+1] == 'L' or ty[i+1] == 'N' or ty[i+1] == 'R' or ty[i+1] == 'V' or ty[i+1] == 'a' or ty[i+1] == 'f' or ty[i+1] == 'j' :
                lst_data.append(temp_lst)

    max_len = 0
    for elem in lst_data:
        if max_len < len(elem):
            max_len = len(elem)
            
    for index in range(len(lst_data)):
        insert = max_len - len(lst_data[index])
        while(insert>0):
            lst_data[index].append(0)
            insert -= 1
            
         
    num = range(len(lst_data))
    random.shuffle(num)
    lst_data = np.array(lst_data)
    label = np.array(label)
    
    lst_data = lst_data[num][:]
    label = label[num][:]
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(label)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    label = onehot_encoder.fit_transform(integer_encoded)

    split_num = int(len(lst_data)*0.8)   
    train_data = lst_data[0:split_num][:]
    test_data = lst_data[split_num:-1][:]
    train_label = label[0:split_num]
    test_label = label[split_num:-1]
    
    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)    
    np.save('train.npy', train_data)
    np.save('test.npy', test_data)
#    train_data = train_data.tolist()
#    test_data = test_data.tolist()    
    
#    #save data
#    writer_tr = csv.writer(file('./tr_data.csv','wb'))
#    writer_te = csv.writer(file('./te_data.csv','wb'))
#    for index_tr in range(len(train_data)):
#        writer_tr.writerow(train_data[index_tr])
#    #writer_tr.close()
    
#    for index_te in range(len(test_data)):
#        writer_te.writerow(test_data[index_te])
    #writer_te.close()    
    np.save('train_label.npy',train_label)
    np.save('test_label.npy',test_label)    
    #return train_data,test_data,train_label,test_label,max_len


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial) 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv1d(x,W):
    return tf.nn.conv1d(x, W, stride=5, padding='VALID')
def max_pool_1x1(x):
    return tf.nn.max_pool(x,[1,1,2,1],[1,1,1,1],'VALID')

def generatebatch(data, label, batch_size):
    for batch_i in range(data.shape[0]/batch_size):
        start = batch_i * batch_size
        end = start + batch_size
        batch_xs = data[start:end,:]
        batch_ys = label[start:end,:]
        batch_xs = np.reshape(batch_xs, [batch_xs.shape[0],batch_xs.shape[1],1])
        #batch_ys = np.reshape(batch_ys, [-1,batch_ys.shape[0],batch_ys.shape[1]])
        
        yield batch_xs, batch_ys


if __name__ == '__main__':

    kernel_size = 3
    epochs = 10
    filters = 64
    hidden_dims = 250
    batch_size = 100
    
    file_lst = ['./mitdata/100','./mitdata/101','./mitdata/102','./mitdata/103',
                './mitdata/104','./mitdata/105','./mitdata/106','./mitdata/107',
                './mitdata/108','./mitdata/109','./mitdata/111','./mitdata/112',
                './mitdata/113','./mitdata/114','./mitdata/115','./mitdata/116',
                './mitdata/117','./mitdata/118','./mitdata/119','./mitdata/121',
                './mitdata/122','./mitdata/123','./mitdata/124','./mitdata/200',
                './mitdata/201','./mitdata/202','./mitdata/203','./mitdata/205','./mitdata/207',
                './mitdata/208','./mitdata/209','./mitdata/210','./mitdata/212','./mitdata/213',
                './mitdata/214','./mitdata/215','./mitdata/217','./mitdata/219','./mitdata/220',
                './mitdata/221','./mitdata/222','./mitdata/223','./mitdata/228','./mitdata/230',
                './mitdata/231','./mitdata/232','./mitdata/233','./mitdata/234']
    
    process_qrs_index(file_lst)  
    
    x_train = np.load('train.npy')
    x_test = np.load('test.npy')
    print x_train.shape
    print x_test.shape

    y_train = np.load('train_label.npy')
    y_test = np.load('test_label.npy')
    x_train = np.reshape(x_train,[x_train.shape[0], x_train.shape[1], 1])
    x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1], 1])
    
    #train_data = np.reshape(train_data, [-1,train_data])
    print x_train.shape
    print('Build model...')
    model = Sequential()
    
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    
    
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=2, input_shape=[2114, 1]))
    # we use max pooling:
    model.add(MaxPooling1D())
    
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1,))
    model.add(GlobalMaxPooling1D())
    
    
    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(11))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    result = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs, shuffle=True,
              validation_data=(x_test, y_test))
              
    
    plt.figure('accuracy image')
    plt.plot(result.epoch,result.history['acc'],label="train_acc")
    plt.plot(result.epoch,result.history['val_acc'],label="val_acc")
    plt.scatter(result.epoch,result.history['acc'],marker='*')
    plt.scatter(result.epoch,result.history['val_acc'],marker='>')
    plt.legend(loc='upper right')



        
    plt.xlabel('Number of epoch')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('NNacc.png',format='png')
    
    plt.figure('loss image')
    plt.plot(result.epoch,result.history['loss'],label="train_loss")
    plt.plot(result.epoch,result.history['val_loss'],label="val_loss")
    plt.scatter(result.epoch,result.history['loss'],marker='*')
    plt.scatter(result.epoch,result.history['val_loss'],marker='>')
    plt.legend(loc='upper left')
    plt.xlabel('Number of epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('NNlos.png',format='png')          
    
    
    
    
    
    
    
    
#    x = tf.placeholder(tf.float32, [None, train_data.shape[1], 1])
#    y = tf.placeholder(tf.float32, [None, 11])
#    
#    ## conv1 layer ##
#    W_conv1 = weight_variable([4, 1, 32]) # patch 5x5, in size 1, out size 32
#    b_conv1 = bias_variable([32])
#    
#    x_input = tf.reshape(x,[-1,train_data.shape[1],1])    
#    h_conv1 = tf.nn.relu(conv1d(x_input, W_conv1) + b_conv1) # output size 28x28x32
#    print 'h_conv1 shape:'
#    print h_conv1.shape
#   
#    ## conv2 layer ##
#    W_conv2 = weight_variable([3, 32, 64]) # patch 5x5, in size 32, out size 64
#    b_conv2 = bias_variable([64])
#    h_conv2 = tf.nn.relu(conv1d(h_conv1, W_conv2) + b_conv2) # output size 14x14x64 
#    print 'h_conv2 shape:'
#    print h_conv2.shape
#    
#    ##conv3 layer
#    W_conv3 = weight_variable([3, 64, 128])
#    b_conv3 = bias_variable([128])
#    h_conv3 = tf.nn.relu(conv1d(h_conv2, W_conv3) + b_conv3)
#    print 'h_conv3 shape:'
#    print h_conv3.shape
#    
#    
#    #a = raw_input()
#    ## func1 layer ##ds
#    W_fc1 = weight_variable([17*128, 1024])
#    b_fc1 = bias_variable([1024])
#    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
#    h_pool2_flat = tf.reshape(h_conv3, [-1, 17*128])
#    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#    h_fc1_drop = tf.nn.dropout(h_fc1, 0.1)
#    
#    ## func2 layer ##
#    W_fc2 = weight_variable([1024, 11])
#    b_fc2 = bias_variable([11])
#    
#    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction),reduction_indices=[1]))
#    train_step = tf.train.AdamOptimizer(1e-1).minimize(cross_entropy)
#
#    y_pred = tf.arg_max(prediction,1)   
#    bool_pred = tf.equal(tf.arg_max(y,1),y_pred)
#    accuracy = tf.reduce_mean(tf.cast(bool_pred,tf.float32))    
#    
#    sess = tf.Session()
#    # important step
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        for epoch in range(20): # 迭代1000个周期
#            for batch_xs, batch_ys in generatebatch(train_data,train_label,100):
#                sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
#            res = sess.run(accuracy, feed_dict = {x:batch_xs, y:batch_ys})
#            print (epoch, res)






