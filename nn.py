#! /usr/bin/env python
#coding=utf-8
from __future__ import division
from functools import reduce
import re
import tarfile
import math

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence

from keras.layers import Input, Embedding, LSTM, Dense, merge, TimeDistributed
from keras.models import Model, Sequential
from sklearn.metrics import average_precision_score
from keras.layers.core import *

from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D

from keras.layers.recurrent import GRU

EMBED_SIZE = 16
HIDDEN_SIZE = 16
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 3


nb_filter=250
filter_length=3

def get_embedding_input_output(part_name,vocab_size):
    main_input = Input(shape=(MAX_LEN,), dtype='int32', name=part_name+'_input')
        
    x = Embedding(output_dim=EMBED_SIZE, input_dim=vocab_size, input_length=MAX_LEN)(main_input)
        
    return main_input,x


def lstm_train(X_train,y_train,vocab_size):
    
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
           
    main_input = Input(shape=(MAX_LEN,), dtype='int32')
           
    x = Embedding(output_dim=EMBED_SIZE, input_dim=vocab_size, input_length=MAX_LEN)(main_input)
           
    lstm_out = LSTM(HIDDEN_SIZE)(x)
    
    main_loss = Dense(1, activation='sigmoid', name='main_output')(lstm_out)
    
    model = Model(input=main_input, output=main_loss)
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS)
        
    return model


def cnn_train(X_train,y_train,vocab_size):
    
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
           
    print('Build model...')
    model = Sequential()
    model.add(Embedding(vocab_size, EMBED_SIZE, input_length=MAX_LEN))
    
    model.add(Dropout(0.25))
    
    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling1D(pool_length=2))
    
    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    model.add(Flatten())
    
    # We add a vanilla hidden layer:
    model.add(Dense(HIDDEN_SIZE))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, show_accuracy=True)
        
    return model

def cnn_combine_train(X_train_list,y_train,vocab_size):
    N=len(X_train_list)
    
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    
    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_embedding_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
            
    x = merge(out_list,mode='concat')
    
    x = Dropout(0.25)(x)
        
    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    x = Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1)(x)
                            
    # we use standard max pooling (halving the output of the previous layer):
    x = MaxPooling1D(pool_length=2)(x)
    
    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    x = Flatten()(x)
    
    # We add a vanilla hidden layer:
    x = Dense(HIDDEN_SIZE)(x)
    x = Dropout(0.25)(x)
    x = Activation('relu')(x)
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(input=input_list, output=x)
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(X_train_list, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS)
    
    return model

def lstm_combine_train(X_train_list,y_train,vocab_size):
    N=len(X_train_list)
    
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    
    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_embedding_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
            
    x = merge(out_list,mode='concat')
    
    x = LSTM(HIDDEN_SIZE)(x)
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(input=input_list, output=x)
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(X_train_list, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS)
    
    return model

def lstm_attention_combine_train(X_train_list,y_train,vocab_size):
    N=len(X_train_list)
    
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    
    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_embedding_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
            
    x = merge(out_list,mode='concat')
    
    lstm_out = LSTM(HIDDEN_SIZE, return_sequences=True)(x)
    
    x = lstm_out
    for i in range(10):
        att = TimeDistributed(Dense(1))(x)
        att = Flatten()(att)
        att = Activation(activation="softmax")(att)
        att = RepeatVector(HIDDEN_SIZE)(att)
        att = Permute((2,1))(att)
        x = att

    mer = merge([att, lstm_out], "mul")
    mer = merge([mer, out_list[-1]], 'mul')
    hid = AveragePooling1D(pool_length=2)(mer)
    hid = Flatten()(hid)
    
    #hid = merge([hid,out_list[-1]], mode='concat')
        
    main_loss = Dense(1, activation='sigmoid', name='main_output')(hid)
    
    model = Model(input=input_list, output=main_loss)
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(X_train_list, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS)
    
    return model


def lstm_memory_train(X_train_list,y_train,vocab_size):
    N=len(X_train_list)
    
    X_train_list = [sequence.pad_sequences(x_train, maxlen=MAX_LEN) for x_train in X_train_list]
    
    input_list=[]
    out_list=[]
    for i in range(N):
        input,out=get_embedding_input_output('f%d' %i,vocab_size)
        input_list.append(input)
        out_list.append(out)
            
    x = merge(out_list,mode='concat')
    
    lstm_out = LSTM(HIDDEN_SIZE, return_sequences=True)(x)
    
    lstm_share=GRU(HIDDEN_SIZE, return_sequences=True)
    
    x = lstm_out
    for i in range(2):
        att = TimeDistributed(Dense(1))(x)
        att = Flatten()(att)
        att = Activation(activation="softmax")(att)
        att = RepeatVector(HIDDEN_SIZE)(att)
        att = Permute((2,1))(att)
        
        mer = merge([att, lstm_out], "mul")
        mer = merge([mer, out_list[-1]], 'mul')
        
        z = merge([lstm_out,mer],'sum')
        z = lstm_share(z)
        x = z

    hid = AveragePooling1D(pool_length=2)(x)
    hid = Flatten()(hid)
    
    #hid = merge([hid,out_list[-1]], mode='concat')
        
    main_loss = Dense(1, activation='sigmoid', name='main_output')(hid)
    
    model = Model(input=input_list, output=main_loss)
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(X_train_list, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS)
    
    return model
