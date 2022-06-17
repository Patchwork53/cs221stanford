from pickletools import read_uint8
import sys
sys.setrecursionlimit(100000)

class TransportationProblem(object):
    def __init__(self, N) -> None:
        self.N = N
    
    def startState(self):
        return 1
    
    def isEnd(self, state):
        return state == self.N
    
    def succAndCost(self, state):
        result =[]
        if state+1<=self.N:
            result.append(('walk',state+1,1))
        if state*2<=self.N:
            result.append(('tram',state*2,2))
        
        return result


class TransportationProblem2(object):
    def __init__(self, N, weights) -> None:
        self.N = N
        self.weights = weights
    
    def startState(self):
        return 1
    
    def isEnd(self, state):
        return state == self.N
    
    def succAndCost(self, state):
        result =[]
       
        if state*2<=self.N:
            result.append(('tram',state*2, self.weights['tram']))
        if state+1<=self.N:
            result.append(('walk',state+1, self.weights['walk']))
        
        return result




def backtrackingSearch(problem):
    best ={
        'cost': float('+inf'),
        'history': None
    }
    def recurse(state, history, totalCost):
      
        if problem.isEnd(state):
            if totalCost<best['cost']:
                best['cost']=totalCost
                best['history']=history
        
        for action, nextState, cost in problem.succAndCost(state):
            recurse(nextState, history+[(action, nextState, cost)] , totalCost+cost)
    
    recurse(problem.startState(), [], 0)

    return best


def DP(problem):

    #cache for each state which contains future_cost and path_to_end
    #cache updated when recurse returns something

    cache = [None]*(problem.N+1)

    def recurse(state):

        if problem.isEnd(state):
            #total_cost, travel to state with cost
            return {'cost':0, 'path':[]}
        
        if cache[state]!= None:
            return cache[state]

        best = {
            'cost': float('+inf'),
            'path' : None
        }

        for action, nextState, cost in problem.succAndCost(state):
            
            nxt = recurse(nextState)
            
            if nxt['cost']+cost < best['cost']:
                best['cost'] = nxt['cost']+cost
                best['path'] = [(action,str(nextState),str(cost))] + nxt['path']
        

        cache[state] = best

        return best
            
    return recurse(1)


def dynamicProgramming(problem):
    cache = [None]*(problem.N+1)


    def futureCost(state):
        if problem.isEnd(state):
            return 0
        
        if cache[state]!=None: 
            return cache[state]
             
        result = min ( (cost+futureCost(newState)) 
                        for action, newState, cost in problem.succAndCost(state))
        cache[state] = result
        
        return result
    
    return (futureCost(problem.startState()),[])
     
    
     
    
class minPriorityQueue(object):

    def __init__(self):
        self.arr = [None]*2000

    def insert (self, state, cost):

        if self.arr[1]==None:
            self.arr[1] = [state, cost]
            self.top = 2 
            return
        
        self.arr[self.top] = [state, cost]
        n = self.top

        self.top = 2 
        while self.arr[self.top]!=None:
            self.top+=1
        while n!=1 :
            half = int(n/2)
            
            if self.arr[half][1] > self.arr[n][1] :
                temp = self.arr[half] 
                self.arr[half] = self.arr[n]
                self.arr[n] = temp
                n=half
            else:
                break
    
    def update (self, state, newCost):
        #only ever reduced
        n = -1
        for i in range(len(self.arr)):
            if self.arr[i] == None:
                continue

            if self.arr[i][0] == state:
                n = i
                if self.arr[i][1] < newCost:
                    return

                self.arr[i][1] = newCost

                while n!=1 :
                    half = int(n/2)
                    if self.arr[half][1] > self.arr[n][1] :
                        temp = self.arr[half] 
                        self.arr[half] = self.arr[n]
                        self.arr[n] = temp
                        n=half
                    else:
                        break
        if n == -1:
            self.insert(state, newCost)
    
    def extractMin(self):

        result = self.arr[1]
        n = 1
        while True:
            if self.arr[2*n] ==None and self.arr[2*n+1]==None:
                self.arr[n] = None
                self.top=n
                break
            
            elif  self.arr[2*n+1]==None:
                self.arr[n] = self.arr[2*n]
                n = 2*n
            
            elif  self.arr[2*n]==None:
                self.arr[n] = self.arr[2*n+1]
                n = 2*n+1
            

            elif  self.arr[2*n][1] <=  self.arr[2*n+1][1]:
                self.arr[n] = self.arr[2*n]
                n = 2*n
            else:
                self.arr[n] = self.arr[2*n+1]
                n = 2*n+1

        return result

    def print(self):
        print(self.arr)


def uniformCostSearch(problem):

    Q = minPriorityQueue()
    Q.insert(problem.startState(), 0)
    while True:
        state, totalCost = Q.extractMin()
        # print(state,totalCost)
        if problem.isEnd(state):
            return totalCost

        for action, nextState, cost in problem.succAndCost(state):
            # print("IN", nextState, cost)
            Q.update(nextState, totalCost+cost) 

# print(DP(TransportationProblem(1000)))

def predict(N, weights):
    problem = TransportationProblem2(N, weights)
    history = DP(problem)['path']
    return [action for action, state, cost in history]

def generate_examples():
    true_weights = {'walk':1, 'tram':2}
    return [(N, predict(N,true_weights)) for N in range(1,20)  ]

def structuredPerceptron(examples):
    weights = {'walk':0, 'tram':0}

    for t in range(100):

        for N,real_path in examples:

            guess_path = predict(N,weights)
            for action in real_path:
                weights[action]-=1
            for action in guess_path:
                weights[action]+=1

    return weights


# examples = generate_examples()
# for example in examples:
#     print(example)
# print(structuredPerceptron(examples))
print(dynamicProgramming(TransportationProblem(960)))