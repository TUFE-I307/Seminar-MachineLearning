# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:57:34 2019

@author: 韩琳琳
"""

###################鸢尾花实验

import numpy as np
import pandas as pd
import random
import os
os.chdir('C:\\Users\\SA\\Documents\\machine learning\\code\\Ch04')

dataset=pd.read_csv('iris.txt',header=None)


#切分训练级和测试集
def randsplit(dataset,rate):
    l=list(dataset.index)
    random.shuffle(l)
    dataset.index=l
    n=dataset.shape[0]
    m=int(n*0.8)
    train=dataset.loc[range(m),:] #注意不能写成.iloc or [:m,:]
    test=dataset.loc[range(m,n),:]
    test.index=range(n-m)
    dataset.index=range(n)
    return train,test
train,test=randsplit(dataset,0.8)

#构建贝叶斯分类器
def gauss_clsssify(train,test):
    labels=train.iloc[:,-1].value_counts().index 
    mean=[]
    std=[]
    for la in labels:
        item=train.loc[train.iloc[:,-1]==la] # loc 列名不能为-1
        m=item.iloc[:,:-1].mean()
        s=np.sum((item.iloc[:,:-1]-m)**2)/item.shape[0]
        mean.append(m)
        std.append(s)
    mean=pd.DataFrame(mean) #mean=pd.DataFrame(mean,index=labels)
                            #这样后面的 pla=p.index[np.argmax(p.values)]
    std=pd.DataFrame(std)
    result=[]
    
    for i in range(test.shape[0]):
        iest=test.iloc[i,:-1].tolist() #当前测试实例
        pr=np.exp(-(iest-mean)**2/(2*std))/np.sqrt(2*np.pi*std) #得到正态分布概率矩阵
        p=1
        for j in range(test.shape[1]-1):
            p*=pr[j]
            pla=labels[p.index[np.argmax(p.values)]]
        result.append(pla)
    test['pre']=result
    accuracy=(test.iloc[:,-2]==test.iloc[:,-1]).mean()
    print(f'模型的预测准确率为{accuracy}')
    

for i in range(20):
    train ,test=randsplit(dataset,0.8)
    gauss_clsssify(train,test)
    


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import datasets
iris=datasets.load_iris()

#切分数据集
Xtrain,Xtest,ytrain,ytest=train_test_split(iris.data,iris.target,random_state=42)#随机数种子决定不同切分规则
#建模
clf=GaussianNB()
clf.fit(Xtrain,ytrain)
#在测试集上执行预测，proba导出的是每个样本属于某一类的概率
clf.predict(Xtest)
clf.predict_proba(Xtest)
#测试准确率
accuracy_score(ytest,clf.predict(Xtest))

#连续性用高斯贝叶斯，0-1用伯努利贝叶斯，分词用多项式朴素贝叶斯




#################################朴素贝叶斯之言论过滤

import numpy as np

def loadDataSet():
    dataset=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 侮辱性词表, 0 非侮辱性词表
    return dataset,classVec

dataset,classVec = loadDataSet()
                 
#构建词汇表

def creatVocabList(dataset):
    vocablist=set()     #只有set和set才能取并集
    for doc in dataset:
        vocablist=vocablist|set(doc) #并集
    vocablist=list(vocablist) 
    #vocablist=set(vocablist)   #并集的结果已经去过重了
    return vocablist

vocablist = creatVocabList(dataset)

#获得训练集向量

def setOfWords2Vec(vocablist,inputset): #输入词表和切分好的一个词条
    returnVec = [0]*len(vocablist)    #与词表等长的零向量
    for word in inputset:
        if word in vocablist:
            returnVec[vocablist.index(word)] = returnVec[vocablist.index(word)]+1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def get_trainMat(dataset):
    vocablist=creatVocabList(dataset)
    result=[]
    for inputset in dataset:
        vec=setOfWords2Vec(vocablist,inputset)
        result.append(vec)
    return result

trainMat = get_trainMat(dataset)

#朴素贝叶斯分类器训练函数

def trainNB(trainMat,classVec):
    n = len(trainMat)   #总文档数目
    m = len(trainMat[0]) #所有文档中非重复词条数
    pA0 = sum(classVec)/n  #侮辱性文档占总文档的概率
    p0num = np.zeros(m)   # 初始化
    p1num = np.zeros(m)
    p1demo = 0
    p0demo = 0
    for i in range(n):
        if classVec[i]==1:       
            p1num += trainMat[i] # 侮辱性文档中词条的分布
            p1demo += sum(trainMat[i]) #侮辱性文档中词条总数
        else:
            p0num += trainMat[i]
            p0demo += sum(trainMat[i])
    p1v = p1num/p1demo       #全部侮辱类词条的条件概率数组
    p0v = p0num/p0demo 
    return p1v,p0v,pA0 

p1v,p0v,pA0 = trainNB(trainMat,classVec)

#测试朴素贝叶斯分类器
 
from functools import reduce

def classifyNB(vec2classify,p1v,p0v,pA0): # vec2classify 待分类的词条分布数组
    p1 = reduce(lambda x,y:x*y,vec2classify*p1v)*pA0  #reduce作用，对应数字相乘(已知词组属于侮辱类的条件概率*pA0 )
    p0 = reduce(lambda x,y:x*y,vec2classify*p0v)*(1-pA0)
    print('p1:',p1)
    print('p0:',p0)
    if p1>p0:
        return 1
    else:return 0

#朴素贝叶斯测试函数

def testingNB(testVec):
    dataset,classVec = loadDataSet()
    vocablist = creatVocabList(dataset)
    trainMat = get_trainMat(dataset)
    p1v,p0v,pA0 = trainNB(trainMat,classVec)
    thisone = setOfWords2Vec(vocablist,testVec)
    if classifyNB(thisone,p1v,p0v,pA0) == 0:
        print(testVec,'属于非侮辱类') 
    else:
        print(testVec,'属于侮辱类') 
        
testVec1 = ['love','my','dalmation']
testingNB(testVec1)
testVec2 = ['garbage','dog']
testingNB(testVec2)

###################朴素贝叶斯改进之拉普拉斯平滑 
#问题1 ： P(W0|1)P(W1|1)P(W2|1) 其中任何一个为0，乘积也为0
#解决 ：拉普拉斯平滑：将所有词的初始频数设为1，分母设为2
#问题2 ： P(W0|1)P(W1|1)P(W2|1) 每个都太小，数据下溢出
#解决 ： 对乘积结果取对数

#朴素贝叶斯分类器训练函数 改进版

def trainNB2(trainMat,classVec):
    n = len(trainMat)   #总文档数目
    m = len(trainMat[0]) #所有文档中非重复词条数
    pA0 = sum(classVec)/n  #侮辱性文档占总文档的概率
    p0num = np.ones(m)   # 初始化 1
    p1num = np.ones(m)
    p1demo = 2         #分母设为2
    p0demo = 2
    for i in range(n):
        if classVec[i]==1:       
            p1num += trainMat[i] # 侮辱性文档中词条的分布
            p1demo += sum(trainMat[i]) #侮辱性文档中词条总数
        else:
            p0num += trainMat[i]
            p0demo += sum(trainMat[i])
    p1v = np.log(p1num/p1demo)  #侮辱类的条件概率数组取对数
    p0v = np.log(p0num/p0demo) 
    return p1v,p0v,pA0 

p1v,p0v,pA0 = trainNB2(trainMat,classVec)

#测试朴素贝叶斯分类器
 
from functools import reduce

def classifyNB2(vec2classify,p1v,p0v,pA0): # vec2classify 待分类的词条分布数组
    p1 = sum(vec2classify*p1v)+np.log(pA0)  # 原本的连乘取对数变成连加
    p0 = sum(vec2classify*p0v)+np.log(1-pA0)
    print('p1:',p1)
    print('p0:',p0)
    if p1>p0:
        return 1
    else:return 0

#朴素贝叶斯测试函数

def testingNB2(testVec):
    dataset,classVec = loadDataSet()
    vocablist = creatVocabList(dataset)
    trainMat = get_trainMat(dataset)
    p1v,p0v,pA0 = trainNB2(trainMat,classVec)
    thisone = setOfWords2Vec(vocablist,testVec)
    if classifyNB2(thisone,p1v,p0v,pA0) == 0:
        print(testVec,'属于非侮辱类') 
    else:
        print(testVec,'属于侮辱类') 


