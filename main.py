#! /usr/bin/env python
#coding=utf-8
from document import *
from pmi import pmi_prediction
from evaluate import eval
from bigram import bigram_prediction
from classify import *
import nn
from keras.preprocessing import sequence

train=read_question_corpus(train_file)
test=read_c_and_j_corpus()

train=train[:10000]

results=pmi_prediction(train,test)
eval(test,results)

#results=bigram_prediction(train,test)
#eval(test,results)
test0=test
train,test=get_question_documents(train),get_question_documents(test0)
V=get_vocabrary(train,k=1000)

# pair wise
train_x,train_y=get_pair_wise_documents_for_train(train,V)
test_x,test_y=get_pair_wise_documents(test,V)

v_len=len(V)*2
model=nn.lstm_train(train_x,train_y,v_len)

test_x = sequence.pad_sequences(test_x, maxlen=nn.MAX_LEN)
X_pred = model.predict(test_x)

index=0
results=[]
for documents in test:
    choice_results=[]
    for j,choice_documents in enumerate(documents):
        score=0
        for label,choice_features,context_features in choice_documents:
            score+=X_pred[index][0]
            index+=1
        choice_results.append((score,j))
    choice_results.sort()
    results.append(choice_results[-1][1])

eval(test0,results)
