from doctest import OutputChecker
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.special import rel_entr

def mms(ypre):
    return (ypre - np.min(ypre))/(np.max(ypre) - np.min(ypre))
    #return (ypre - np.mean(ypre))/np.std(ypre)#(np.max(ypre) - np.min(ypre))

def sigmoid(x,a,b): return 1/(1+np.exp(-a*(x))) #-b

def determine(x,ruler,pr =np.array([0.5,0.7,0.9])):
    y = pd.cut(x,ruler,labels = False)
    y[np.isnan(y)] = 0
    return pr[y.astype(int)]
    


def PickUp(ReadLimit,Index,age,M,N,beta0,beta1,beta2,beta3):
    p = 1/(1+np.exp(beta0+beta1*M+beta2*N+beta3*age))
    p = p/sum(p)
    
    if sum(p > 0) < ReadLimit:
        num = sum(p>0)
        _ = Index[p>0].tolist()
        _ += np.random.choice(Index[p==0],ReadLimit-num,replace=False).tolist()
        return np.array(_)
    
    return np.random.choice(Index,ReadLimit,p=p,replace=False)

def Simulation(IQ,age,Concise,NReaders,ReadLimitMean,ReadLimitSD,beta0,beta1,beta2,beta3,a,SocialInfluence):
    try:
        #########Initialization###########
        Mean = np.mean(IQ)
        STD = np.std(IQ)
        Index = np.arange(len(age))
        M = np.zeros(len(IQ))
        N = np.zeros(len(IQ))

        #Concise = mms(Concise)
        ruler = np.quantile(Concise,[0,0.45,0.85,1])
        ##########
        for i in (range(NReaders)):
            
            ###################Pick-up stage#################
            ReadLimit = np.max([1,int(np.random.normal(ReadLimitMean,ReadLimitSD))]) 
            ReadReviews = PickUp(ReadLimit,Index,age,M,N,beta0,beta1,beta2,beta3)

            Threshold = np.random.normal(Mean,STD)
            ###########################

            theta = IQ[ReadReviews] >= Threshold
            #HelpfulReviews = ReadReviews[theta]

            #if len(HelpfulReviews):
                #b = np.quantile(Concise,0.25)
                #qi = sigmoid(Concise[HelpfulReviews],a,b)

            #    qi = determine(Concise,ruler)[HelpfulReviews]
                        
            #    Alpha = (SocialInfluence*qi).astype(int)#*theta.astype(int) #np.random.binomial((SocialInfluence*N[HelpfulReviews]).astype(int),qi)
            #    Beta =  SocialInfluence-Alpha #(SocialInfluence*N[HelpfulReviews]).astype(int) - Alpha
            
            #    votepro = np.random.beta(1+Alpha+M[HelpfulReviews]\
            #            ,1+Beta+N[HelpfulReviews]-M[HelpfulReviews]) 
                       
            #    VoteReviews = HelpfulReviews[np.random.binomial(1,p=votepro).astype(bool)]
            
            #    if len(VoteReviews): M[VoteReviews] += 1
            
            #b = np.quantile(Concise,0.25)
            #qi = sigmoid(Concise[ReadReviews],a,b)
            
            qi = determine(Concise,ruler)[ReadReviews]
            
            Alpha = (SocialInfluence*qi).astype(int)*theta.astype(int)
            Beta =  SocialInfluence-Alpha
            votepro = np.random.beta(1+Alpha+M[ReadReviews]\
                        ,1+Beta+N[ReadReviews]-M[ReadReviews]) 
            VoteReviews = ReadReviews[np.random.binomial(1,p=votepro).astype(bool)]
            if len(VoteReviews): M[VoteReviews] += 1
                
            N[ReadReviews] += 1
    except Exception as errI:
        print(errI)


    return M, N

def SocialLearning(IQ,age,NReaders,ReadLimitMean,ReadLimitSD,beta0,beta1,beta2,beta3):
    #########Initialization###########

    Index = np.arange(len(age))
    M = np.zeros(len(IQ))
    N = np.zeros(len(IQ))
    ##########

    for i in (range(NReaders)):
        ###################Pick-up stage#################
        ReadLimit = np.max([1,int(np.random.normal(ReadLimitMean,ReadLimitSD))])
        ReadReviews = PickUp(ReadLimit,Index,age,M,N,beta0,beta1,beta2,beta3)

    
        votepro = np.random.beta(1+M[ReadReviews]\
                ,1+N[ReadReviews]-M[ReadReviews]) 

        VoteReviews = ReadReviews[np.random.binomial(1,p=votepro).astype(bool)]
        if len(VoteReviews):M[VoteReviews] += 1

        N[ReadReviews] += 1
    return M, N


def IndependentVote(IQ,age,Concise,NReaders,ReadLimitMean,ReadLimitSD,beta0,beta1,beta2,beta3,a):
    #########Initialization###########
    Mean = np.mean(IQ)
    STD = np.std(IQ)
    Index = np.arange(len(age))
    M = np.zeros(len(IQ))
    N = np.zeros(len(IQ))
    
    #Concise = mms(Concise)
    ruler = np.quantile(Concise,[0,0.45,0.85,1])
    ##########

    for i in (range(NReaders)):
        ###################Pick-up stage#################
        ReadLimit = np.max([1,int(np.random.normal(ReadLimitMean,ReadLimitSD))])
        ReadReviews = PickUp(ReadLimit,Index,age,M,N,beta0,beta1,beta2,beta3)

        Threshold = np.random.normal(Mean,STD) #np.quantile(IQ,0.75)#
        ###########################

        theta = IQ[ReadReviews] >= Threshold
        #HelpfulReviews = ReadReviews[theta]

        #if len(HelpfulReviews):
            #b = np.quantile(Concise,0.25)
            #qi = sigmoid(Concise[HelpfulReviews],a,b)
        #    qi = determine(Concise,ruler)[HelpfulReviews]
        
        #    Indicator = np.random.binomial(1,p=qi)#*theta
        #    VoteReviews = HelpfulReviews[Indicator.astype(bool)]
        
        #    if len(VoteReviews):M[VoteReviews] += 1

        #b = np.quantile(Concise,0.25)
        #qi = sigmoid(Concise[ReadReviews],a,b)
        
        qi = determine(Concise,ruler)[ReadReviews]
        
        Indicator = np.random.binomial(1,p=qi)*theta.astype(int)
        VoteReviews = ReadReviews[Indicator.astype(bool)]
        
        if len(VoteReviews):M[VoteReviews] += 1

        N[ReadReviews] += 1
    return M, N

def MP_Simualtion(Param):
    Num,Helpful,IQ,age,Concise,NReaders,ReadLimitMean,ReadLimitSD,beta0,beta1,beta2,beta3,a,SocialInfluence = Param

    M = np.zeros((Num,len(IQ)))
    N = np.zeros((Num,len(IQ)))

    for num in range(Num):
        m,n = Simulation(IQ,age,Concise,NReaders,ReadLimitMean,ReadLimitSD,beta0,beta1,beta2\
                                        ,beta3,a,SocialInfluence)
        M[num] = m
        N[num] = n

    ypre = np.mean(np.log1p(M),axis=0)
                
    corr = pearsonr(Helpful,ypre)[0]
    corr_pvalue = pearsonr(Helpful,ypre)[1]
    r2 = r2_score(Helpful/np.sum(Helpful),ypre/np.sum(ypre))
    KLD = np.sum(rel_entr(Helpful+1/np.sum(Helpful+1),ypre+1/np.sum(ypre+1)))

    return((NReaders,a,SocialInfluence,corr,corr_pvalue,r2,KLD,M.tolist(),N.tolist(),Helpful.tolist())) 