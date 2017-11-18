#! /usr/bin/env python
#coding=utf-8
from stopword import StopWord
import os
import random

test_dir = r'chains-test'
train_file = r'../data/train_chains_150k_random'

stop_words=StopWord()

class Event:
    def __init__(self,verb,verb_lemma,sbj,obj):
        self.verb=verb
        self.verb_lemma=verb_lemma
        self.sbj=sbj
        self.obj=obj
        self.iobj=''
        
class Question:
    def __init__(self,answer,context,choices):
        self.answer=answer
        self.context=context
        self.choices=choices

def parse_entity(entity_str):
    entity=[]
    parts=entity_str.split('<&>')
    for part in parts:
        for token in part.lower().split():
            if stop_words.is_in(token)==False:
                entity.append(token)
    return entity
                

def parse_event(event_str):
    parts=event_str.split('<|>')
    verb=parts[0]
    verb_lemma=parts[1]
    sbj=parse_entity(parts[2])
    obj=parse_entity(parts[3])
    
    event=Event(verb,verb_lemma,sbj,obj)
    
    if len(parts[4])>0:
        event.iobj=parse_entity(parts[4].split('+')[1])
    
    return event

def read_question_corpus(path):
    items=[]
    for line in open(path,'rb'):
        line=line.strip()
        if len(line)>0:
            parts=line.split('<@>')
            answer=int(parts[0])
            context=[parse_event(parts[i]) for i in range(1,len(parts)-5)]
            choices=[parse_event(parts[i]) for i in range(len(parts)-5,len(parts))]
            items.append(Question(answer,context,choices))
    return items

def build_question(chains,all_verbs):
    answer=0
    context=chains[:-1]
    choices=[]
    choices.append(chains[-1])
    for v in all_verbs:
        if len(choices)<5:
            if v!=chains[-1].verb_lemma:
                choices.append(Event(v,v,'',''))
        else:
            break
    return Question(answer,context,choices)

def read_c_and_j_corpus():
    documents=[]
    all_verbs=[]
    for fpath in os.listdir(test_dir):
        chains=[]
        for line in open('%s/%s' %(test_dir,fpath),'rb'):
            line=line.strip()
            if len(line)>0:
                p=line.split()
                if len(p)==4:
                    chains.append(Event(p[1],p[2],'','')) # verb,verb_lemma
                    all_verbs.append(p[2])
            else:
                break
        documents.append(chains)
    
    all_verbs=[v for v in set(all_verbs)]
    
    questions=[]
    for chains in documents:
        if len(chains)<9:
            random.shuffle(all_verbs)
            questions.append(build_question(chains,all_verbs))
        else:
            for i in range(0,len(chains)-9):
                random.shuffle(all_verbs)
                questions.append(build_question(chains[i:i+9],all_verbs))
    
    print 'len of c&j questions: %d' %len(questions)
    return questions