#! /usr/bin/env python
#coding=utf-8
from __future__ import division
import math

# Bigram probabilities
# Jans et al., (EACL-2012)

class Bigram:
    def __init__(self,train_questions):
        self.V={}
        self.N=0
        for q in train_questions:
            verb_list=[event.verb_lemma for event in q.context]
            verb_list.append(q.choices[q.answer].verb)
        
            self.N+=len(verb_list)
        
            for i in range(len(verb_list)):
                for j in range(i+1,len(verb_list)):
                    verb_a,verb_b=verb_list[i],verb_list[j]
                    self.add_to_v(verb_a,verb_b)
                    self.add_to_v(verb_b,verb_a)
        
        self.score_d={}

    def add_to_v(self,verb_a,verb_b):
        if verb_a not in self.V:
            self.V[verb_a]={}
        if verb_b not in self.V[verb_a]:
            self.V[verb_a][verb_b]=0
        self.V[verb_a][verb_b]+=1

    def get_score(self,verb_a,verb_b):
        if verb_a in self.V and verb_b in self.V and verb_b in self.V[verb_a]:
            if (verb_a,verb_b) not in self.score_d:
                p_a_b=self.V[verb_a][verb_b]
                p_b=sum(self.V[verb_b].values())
                #return math.log(self.N*p_a_b/(p_a*p_b))
                
                # p(a|b)=C(a&b)/C(b)
                score=p_a_b/(p_b)
                
                self.score_d[(verb_a,verb_b)]=score
                
                return score
            else:
                return self.score_d[(verb_a,verb_b)]
        else:
            return 0
    
    def get_most_similar_choice(self,test_question):
        results=[]
        for i in range(len(test_question.choices)):
            similarity=sum([self.get_score(context_event.verb_lemma,test_question.choices[i].verb_lemma) for context_event in test_question.context])
            results.append((similarity,i))
        results.sort()
        return results[-1][1]


def bigram_prediction(train,test):
    b=Bigram(train)
    
    results=[]
    for test_question in test:
        choice_index=b.get_most_similar_choice(test_question)
        results.append(choice_index)

    return results


    
