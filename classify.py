#! /usr/bin/env python
#coding=utf-8
import numpy as np

def get_vocabrary(question_documents,k=1000):
    # DF
    df={}
    for documents in question_documents:
        for choice_documents in documents:
            for label,choice_features,context_features in choice_documents:    
                for w in choice_features:
                    if w not in df:
                        df[w]=0
                    df[w]+=1
                for w in context_features:
                    if w not in df:
                        df[w]=0
                    df[w]+=1
            
    df=sorted([(df[w],w) for w in df])
    df.reverse()
    df=[w for count,w in df[:1000]]
        
    #print df[:100]

    # feature selection
    V={}
    other_count=0
    for i,w in enumerate(df):
        V[w]=len(V)

    print 'length of V:',len(V)

    return V

def get_features(event):
    words={}
    words['VERB_%s' %event.verb_lemma]=1
    for token in event.sbj:
        words['SBJ_%s' %token]=1
    for token in event.obj:
        words['OBJ_%s' %token]=1
    return words

def get_question_documents(questions):
    question_documents=[]
    for q in questions:
        documents=[]
        for i,choice_event in enumerate(q.choices):
            choice_features=get_features(choice_event)
            if i==q.answer:
                label=1
            else:
                label=0
            choice_documents=[(label,choice_features,get_features(context_event)) for context_event in q.context]
            documents.append(choice_documents)
        question_documents.append(documents)
    return question_documents

def formatK_pair(choice_features,context_features,V):
    x=[]
    for w in choice_features:
        if w in V:
            if choice_features[w]>=1:
                x.append(V[w])
    for w in context_features:
        if w in V:
            if context_features[w]>=1:
                x.append(len(V)+V[w])
    return x

def formatK(words,V):
    x=[]
    for w in words:
        if w in V:
            if words[w]>=1:
                x.append(V[w])
    return x

def get_pair_wise_documents(question_documents,V):
    vec_size=len(V)
    X=[]
    Y=[]
    for documents in question_documents:
        for choice_documents in documents:
            for label,choice_features,context_features in choice_documents:
                X.append(formatK_pair(choice_features,context_features,V))
                Y.append(label)
    return X,Y

def get_pair_wise_documents_for_train(question_documents,V):
    vec_size=len(V)
    X=[]
    Y=[]
    for documents in question_documents:
        neg_count=0
        for choice_documents in documents:
            cLabel=choice_documents[0][0]
            if cLabel==1 or neg_count<1:
                for label,choice_features,context_features in choice_documents:
                    X.append(formatK_pair(choice_features,context_features,V))
                    Y.append(label)
                    break
            
            if cLabel==0:
                neg_count+=1
                
    return X,Y

def get_sequence_documents(question_documents,V):
    X_list=[[] for i in range(len(question_documents[0][0])+1)]
    Y=[]
    for documents in question_documents:
        for choice_documents in documents:
            Y.append(choice_documents[0][0]) # label
            for i,(label,choice_features,context_features) in enumerate(choice_documents):
                X_list[i].append(formatK(context_features,V))
            X_list[-1].append(formatK(choice_documents[0][1],V))
    return X_list,Y

def get_sequence_documents_for_train(question_documents,V):
    X_list=[[] for i in range(len(question_documents[0][0])+1)]
    Y=[]
    for documents in question_documents:
        neg_count=0
        for choice_documents in documents:
            label=choice_documents[0][0]
            if label==1:
                Y.append(label) 
                for i,(label,choice_features,context_features) in enumerate(choice_documents):
                    X_list[i].append(formatK(context_features,V))
                X_list[-1].append(formatK(choice_documents[0][1],V))
            else:
                if neg_count<1:
                    Y.append(label) 
                    for i,(label,choice_features,context_features) in enumerate(choice_documents):
                        X_list[i].append(formatK(context_features,V))
                    X_list[-1].append(formatK(choice_documents[0][1],V))
                neg_count+=1
    return X_list,Y
