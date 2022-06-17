from unittest import result

from matplotlib.pyplot import fill
from numpy import result_type
import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state==len(self.query)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        if state>=len(self.query):
            return []
        result =[]

        for i in range(state,len(self.query)):
            result.append((i+1 , i+1 , self.unigramCost(self.query[state:i+1])))
        
        return result

        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''
 
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    
    indices = [0]+[action for action in ucs.actions]
    indices.pop()
   
    parts =[query[i:j] for i,j in zip(indices, indices[1:]+[None])]  
    # zz$z$zz
    return ' '.join(parts)
    # END_YOUR_CODE

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (wordsegUtil.SENTENCE_BEGIN, self.queryWords[0])
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state[1]=='END'
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)

        result = []
        nxt_index = self.queryWords.index(state[1]) + 1
        next_word = 'END'
        if nxt_index < len(self.queryWords):
            next_word = self.queryWords[nxt_index]

        fills = self.possibleFills(state[1])
        if len(fills)==0:
            fills.add(state[1])
        
        for possibleFill in fills:
            result.append(    [  possibleFill,  (possibleFill, next_word ),  self.bigramCost(state[0],possibleFill)  ]   )
        return result
        # END_YOUR_CODE


def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    # print(queryWords)
   
    ucs = util.UniformCostSearch(verbose=2)


    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    print(ucs.actions)
    if ucs.actions == None:
        return ' '.join(queryWords)
    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (wordsegUtil.SENTENCE_BEGIN,0)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state[1]>=len(self.query)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)

        result = []
        for i in range(state[1], len(self.query)):
            no_vowels = self.query[state[1]:i+1]
            for fill in self.possibleFills(no_vowels):
                result.append( [fill, (fill,i+1), self.bigramCost(state[0], fill)] )
        
        return result

        # END_YOUR_CODE
class JointSegmentationInsertionProblem2(util.SearchProblem):
    def __init__(self, query, unigramCost, possibleFills):
        self.query = query
        self.unigramCost = unigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state == len(self.query)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)

        result = []
        for i in range(state, len(self.query)):
            no_vowels = self.query[state:i+1]
            for fill in self.possibleFills(no_vowels):
                result.append( [fill, i+1, self.unigramCost(fill)] )
        
        return result

        # END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost,possibleFills))
    print(ucs.actions)
    return ' '.join(ucs.actions)
    # END_YOUR_CODE

def segmentAndInsert2(query, unigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    print('correctly calling V2')
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem2(query, unigramCost,possibleFills))
    print(ucs.actions)
    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################

if __name__ == '__main__':
    shell.main()
