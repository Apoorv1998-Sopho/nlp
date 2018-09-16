import nltk
import random
import operator as op
import re
from functools import reduce
import numpy as np
nltk.download('punkt')

# some perticulat functions
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from scipy.stats import rv_discrete

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
gives out a dictionary and a number that
corresponds to the total number of ngrams
of order n in the text.
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
def Generator(mle):
    sentence = []

    # we want fist nGram sampled to have '<s>'
    while True:
        firstNGram = nextWord(mle)
        if '<s>' in firstNGram: 
            break

        else:
            pass

    # adding the first nGram 
    sentence = firstNGram.split()
    lastWord = ' '.join(sentence[1:])

    # finding which ngram we are working with
    n = len(firstNGram.split())

    # iterating untill we sample an nGram with </s>
    while(True):
        nW = nextWord(mle, lastWord)
        if '<s>' not in nW:
            sentence.append(nW.split()[-1])
            lastWord = ' '.join(\
                       sentence[(len(sentence)-n):])

        if '</s>' in nW:
            if sentence[-1] != '</s>':
                sentence.append(nW.split()[-1])
            return ' '.join(sentence)   


'''
Samples the next Ngram provided the lastngram used.
'''
def nextWord(mle, lastWord=None):
    if lastWord == None:
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




#################################################### 4b
'''
brief: add one smoothing specifically 
        for bigrams
@param: dicB, the bigger dictionary 
         containing bigrams probabilities
@param: dicS, the smaller dictionary 
         containing unigrams probabilities
'''
def add1(dicB, dicS):
    r = {}
    for S1 in dicS.keys():
        for S2 in dicS.keys():
            bigKey = S1+' '+S2
            try:
                r[bigKey] = dicB[bigKey] + 1
            except KeyError:
                r[bigKey] = 1
    return r


'''
brief: Returns a good turing smoothed 
        dictionary of the bigram counts
@param: dicB, the bigger dictionary 
         containing bigrams counts
@param: dicS, the smaller dictionary 
         containing unigrams counts

'''
def goodTuring(dicB, dicS):
    FreqN = freqBuckets(dicB)
    t_counts = totalCounts(dicB)


    # getting the number of unseen bigrams i.e. N_0
    t_bigrams, seen_bigrams = possible_avail(dicS)
    unseen_bigrams = t_bigrams - seen_bigrams
    FreqN[0] = unseen_bigrams

    # calculate the new counts for top 10
    newCountsTop10 = {}
    for n in range(10):
        newCountsTop10[n] = (FreqN[n+1]*(n+1)/float(FreqN[n]))
    return newCountsTop10


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


'''
Total number of bigrams
'''
def totalCounts(dic):
    t_counts = 0
    for key in dic.keys():
        t_counts += dic[key]
    return t_counts















