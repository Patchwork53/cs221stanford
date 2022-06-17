import queue
import wordsegUtil
class JointSegmentationInsertionProblem():
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

corpus ='leo-will.txt'

unigramCost, bigramCost = wordsegUtil.makeLanguageModels(corpus)
possibleFills = wordsegUtil.makeInverseRemovalDictionary(corpus, 'aeiou')


query = 'mgnllthppl'
p = JointSegmentationInsertionProblem(query, bigramCost, possibleFills)

for x in p.succAndCost(p.startState()):
    print(x)