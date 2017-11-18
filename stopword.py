#! /usr/bin/env python
#coding=utf-8

class StopWord:
    def __init__(self):
        self.words={}
        for line in open(r'../data/stopwords.txt','rb'):
            line=line.strip()
            if len(line)>0:
                self.words[line]=0
    
    def is_in(self,word):
        return word in self.words