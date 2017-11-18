#! /usr/bin/env python
#coding=utf-8
from __future__ import division

def eval(test,results):
    acc=0
    for i in range(len(test)):
        if results[i]==test[i].answer:
            acc+=1
    print 'Acc:%s' %(acc/len(test))
    