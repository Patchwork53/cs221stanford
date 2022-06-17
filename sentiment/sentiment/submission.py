#!/usr/bin/python

from ast import Assign
from errno import ESTALE
import random
import collections
import math
import sys
import numpy as np
from scipy import rand

from urllib3 import Retry
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    words = x.split(' ')
    dyct = {}
    for word in words:
        if len(word)<1:
            pass
        if word in dyct:
            dyct[word]+=1
        else:
            dyct[word]=1
    return dyct
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def dot_dictionaries(dict1, dict2):
    y = 0
    for word1 in dict1:
        if word1 in dict2:
            y += dict1[word1]*dict2[word1]

    return y

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    for (x,y) in trainExamples:
        fv = featureExtractor(x)
        for word in fv:
                weights[word]=0

    for t in range(numIters): 
       for (x,y) in trainExamples:
           xhi = featureExtractor(x)
           w_fv = dot_dictionaries(xhi,weights)
           gradient_const = 2*(w_fv-y)
           for word in xhi:
               weights[word] -= eta*gradient_const*xhi[word]


    return weights
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {}
        for word in weights:
            flip = random.randint(0,1)
            if flip==1:
                phi[word]=random.randint(1,5)
           
        y = dot_dictionaries(phi, weights)
        if y >= 0 : y = 1
        else: y = -1
        # END_YOUR_CODE
        return (phi, y)
    x = [generateExample() for _ in range(numExamples)]
    print(x)
    return x

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
    # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        word = x.replace(" ","")
        vector = {}

        for i in range(len(word)-n+1):     
            if word[i:i+n] in vector:
                vector[word[i:i+n]]+=1
            else:
                vector[word[i:i+n]]=1
        return vector

        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.

    how do i evaluate the clusters? 
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    clusters=[]
    all_words = []
    for f_v in examples:
        for word in f_v:
            if word not in all_words:
                all_words.append(word)
    
    
    for i in range(K):
        cluster={}
        for word in all_words:
            cluster[word] = random.randint(0,K) #word count random
        clusters.append(cluster)
    
    '''
    assignments = []
    for i in range(len(examples)):
        assignments.append(random.randint(1,K))
    '''
  
    def find_assignment(vector, clusters):
        min = np.inf
        assignment = -1
        for i in range(len(clusters)):
            dst = 0
            for word in vector:
                if word in clusters[i]:
                    dst += (vector[word]-clusters[i][word])* (vector[word]-clusters[i][word])
                else:
                    dst += vector[word]*vector[word]
            
            if  dst < min:
                assignment = i
                min = dst
        return assignment

    
    

    assignments = np.zeros(len(examples))
    old_assignments = assignments
    old_clusters = clusters

    for t in range(maxIters):
        
        for i in range(len(examples)):
            f_v = examples[i]
            assignments[i] = find_assignment(f_v, clusters)
        # print(assignments)

        # for i in range(len(clusters)):
        #     for word in clusters[i]:
        #         clusters[i][word] =  avg_word_count_given_cluster(examples, i, word)#average of all the examples mapped to this cluster
        # #print(clusters)

        clusters = [{} for _ in range(K)]
       
        mapping_count = [0 for _ in range(K)]

        for i in range(len(examples)):
            j = int(assignments[i]) #assigned cluster index
            mapping_count[j]+=1
            for word in examples[i]:
                if word in clusters[j]:
                    clusters[j][word] += examples[i][word]
                else:
                    clusters[j][word] = 0.0 + examples[i][word]
            
        for i in range(len(clusters)):
            for word in clusters[i]:
                clusters[i][word] /= mapping_count[i]


        if clusters == old_clusters and np.array_equal(old_assignments, assignments):
            break
        
        old_clusters = clusters
        old_assignments = assignments

    
    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)


    print(examples)

    totalCost = 5
    return  (clusters, assignments, totalCost)
  