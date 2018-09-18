'''
Author - Apoorv Agnihotri
Roll No. - 16110020
Collaborator - Abhavya Chandra
'''

import nltk
import random
import operator as op
import math
import re
from functools import reduce
import numpy as np
nltk.download('punkt')

# some perticulat functions
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from scipy.stats import rv_discrete
from scipy.optimize import curve_fit

# get tokenized sentences present in "filename"
def getSents(filename):
    # reading the file and using word_tokenize
    f = open(filename, 'r', encoding="utf8")
    raw = f.read()
    f.close()
    
    """
    remove 1 new line as the dataset has useless
    new lines for no reason, well, for better
    readibility. Anyways.

    will replace a single \n with a space.
    """
    line = re.sub(r"\n(\n*)", r"\1", raw)
    return sent_tokenize(raw)


"""
brief:
Run a word tokenizer on the sentences generated and 
remove any nonaplabetical words. Furter lower the
cases in each word.

params:
takes in a list of list of strings.
"""
def lowerPunct(sentences):
    r =[]
    for sentence in sentences:
        list_words = word_tokenize(sentence)
        r.append([word.lower() for word in \
                  list_words if word.isalpha()])
    return r


"""
Add start and stop symbols to a given set of
sentences.
"""
def addStartStop(sentences):
    r =[]
    for sentence in sentences:
        l = ['<s>']
        l.extend(sentence)
        l.append('</s>')
        r.append(l)
    return r


'''
returns n-gram histogram
'''
def nGramCount(sentences, n):
    dic = {}
    counts = 0
    for sentence in sentences:
        state = []
        for word in sentence:
            state.append(word)
            if len(state) > n:
                state.pop(0)
            if len(state) == n:
                counts += 1
                increaseCount(dic, \
                        ngram=state) 
    return dic


'''
brief: This function tries to increase the
count of a perticular ngram.
'''
def increaseCount(dic, ngram):
    strNgram = " ".join(ngram)
    try:
        dic[strNgram] += 1
    except KeyError:
        dic[strNgram] = 1 
    return


'''
brief: This function will try find the
probability of all the ngrams. with the
given smaller dictionary containin the 
counts, giving out a dictionary.
'''
def MLE(dicB, dicS = None):

    # in the case of unigrams finding the counts
    if dicS==None:
        count = 0
        for v in dicB.values():
            count += v

    r = {}

    for key in dicB.keys():
        if dicS != None:
            newKey = ' '.join(key.split()[:-1])
            count = dicS[newKey]
        r[key] = dicB[key]/float(count) 
    return r


# returns the number of possible ngram counts
def possible_avail(dic):
    return nCr(len(dic.keys()), 2), len(dic.keys()) 


# calculating combinations            
def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer//denom


'''
Generates a sentence with the given MLE

@param: dic the dictionary containg the probabilites
         of the nGrams
@param: n which nGram would we like our generator to use
'''
def Generator(mle, length_req):
    sentence = []
    tries_allowed = 10
    tries_left = tries_allowed

    # we want fist nGram sampled to have '<s>'
    while True:
        firstNGram = nextWord(mle)
        if ('<s>' in firstNGram) and ('</s>' not in firstNGram): 
            break
        else:
            pass

    # adding the first nGram 
    sentence = firstNGram.split()
    lastWord = ' '.join(sentence[1:])

    #if the length of nW 1, lastword = none
    if len(sentence) == 1:
        lastWord = ''

    # finding which ngram we are working with
    n = len(firstNGram.split())
    # print ("nw, lw:", sentence, '|', lastWord)

    # iterating untill we sample an nGram with </s>
    while(tries_left >= 0):
        if len(sentence) > 15: # dont want big sentences
            break
        nW = nextWord(mle, lastWord)
        # print ("nw, lw:", nW, '|', lastWord)

        if '<s>' in nW:
            pass # ignore

        elif '</s>' in nW:
            tries_left -= 1
            # cheking if sentence long enough to be returned
            if len(sentence)-1 >= length_req:
                return ' '.join(sentence[1:])

        else: #'</s>' not present
            sentence.append(nW.split()[-1])
            tries_left = tries_allowed
            lastWord = ' '.join(nW.split()[1:])
    return ' '.join(sentence[1:])


'''
Samples the next Ngram provided the lastngram used.
'''
def nextWord(mle, lastWord=''):
    if lastWord == '':
        # all possiblities
        keys = list(mle.keys())
    
    else:
        keys = [k for k in mle.keys() \
                if (lastWord + ' ' in k)]

    #getting probabilities sane
    pk = np.array([mle[k] for k in keys])
    pk = pk/np.sum(pk)
    
    '''
    checking if there are any possible bigrams, 
    if not returning '<s>'
    '''
    if len(keys) == 0:
        return '</s>'
    
    # sampling with replacement
    word = np.random.choice(keys, 1,\
           replace=True, p=pk)
    return word[0]

#################################################### 4b
'''
computes the probability of a sentence in
log space.
'''
def PrLog(sentence, mle):
    temp = sentence.split()
    pr_log = 0
    for i in range(1, len(temp)):
        try:
            pr_log += np.log(mle[(temp[i-1]+' '+temp[i])])
        except: # KeyError or LogError
            pass
    return (pr_log)
#################################################### 4b


'''
Total number of bigrams
'''
def totalTokens(dic):
    t_counts = 0
    for key in dic.keys():
        t_counts += dic[key]
    return t_counts


'''
brief: Class for finding the new
updated counts after add1smoothing
@param: dicB, bigrams histogram
@param: dicS, unigrams histogram
'''
class Add1Smooth(object):
    def __init__(self, dicB, dicS):
        self.dicB = dicB
        self.dicS = dicS

    # returns the updated count
    def NewCount(self, bs):
        try:
            num = self.dicB[bs] + 1
        except KeyError:
            num = 1
        den = totalTokens(self.dicB) + len(self.dicB)

        # returns (C+1)*N/(N+V) that is new count
        return (len(self.dicB) * num)/float(den)

        
'''
brief: Returns a good turing smoothed 
        bigrams histogram
@param: dicB, bigrams histogram
@param: dicS, unigram histogram
'''
class GoodTuring(object):
    def __init__(self, dicB, dicS):
        self.dicB = dicB
        self.dicS = dicS
        self.FreqN = {}
        self.newCounts = {}

    def Prob(self, bigram):
        total_bigrams = totalTokens(self.FreqN) 
        if bigram not in self.dicB.keys():
            return (self.FreqN[0]/total_bigrams)
        return self.dicB[bigram]/self.dicS[bigram.split()[0]]


    def NewCounts(self, counts=10):
        dicB = self.dicB
        dicS = self.dicS
        self.FreqN = freqBuckets(dicB)
        t_counts = totalTokens(dicB) #tokens

        # getting the number of unseen bigrams i.e. N_0
        t_bigrams, seen_bigrams = possible_avail(dicS)
        unseen_bigrams = t_bigrams - seen_bigrams
        self.FreqN[0] = unseen_bigrams
        unCalculated = []

        # calculate the new FreqBuckets
        for i in range(counts):
            try:
                self.newCounts[i] = self.FreqN[i+1]*\
                (i+1)/float(self.FreqN[i])
            except ZeroDivisionError:
                unCalculated.append(i)
                continue # leave blank.

                # estimate the remaining

        # estimate uncalculated
        def func(x, a, k): # f(x) = a*exp(-kx) == Nc
            return a*(np.exp(-k*x))
        popt = curve_fit(func, \
            list(self.FreqN.keys()),\
            list(self.FreqN.values()))
        for i in unCalculated:
            print (i)
            self.newCounts[i] = self.FreqN[i+1]*(i+1)\
            /func(i, popt[0], popt[1])

        return self.newCounts

'''
Getting freq buckets.
'''
def freqBuckets(dic):
    FreqN = {}
    for key in dic.keys():
        freq = dic[key]
        try:
            FreqN[freq] += 1

        except KeyError:
            FreqN[freq] = 1
    FreqN[0]=0
    return FreqN

def perPlexity(sentences, mle, model):
    perp = 0
    for sent in sentences: # all sentences
        prob_sent = 0
        bigrams = nGramCount([sent], 2)
        for bigram in bigrams:
            new_pr = model.Prob(bigram)
            prob_sent = prob_sent + math.log(new_pr)
            
        perp = perp+prob_sent
        perp *= -1/len(sentences)
       
    return math.exp(perp)