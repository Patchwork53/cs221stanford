
from urllib3 import Retry


class TransportationMDP(object):
    def __init__(self, N) -> None:
        self.N = N

    def startState(self):
        return 1

    def isEnd(self, state):
        return state == self.N

    def actions(self, state):
        result = []
        if state+1 <= self.N:
            result.append('walk')
        if state*2 <= self.N:
            result.append('tram')
        return result

    def succProbReward(self, state, action):
        #return (newState, prob, reward)
        result = []

        if action == 'walk':
            result.append((state+1, 1., -1.))

        elif action=='tram':
            result.append((state*2, 0.5,-2.))
            result.append((state, 0.5, -2.))

        return result
    
    def discount(self):
        return 1
    
    def states(self):
        return range(1, self.N+1)


def valueIteration(mdp):
    V = {}
    for state in mdp.states():
        V[state] = 0.

    def Q(state, action):
        return sum( prob*(reward + mdp.discount()*V[nextState]) for nextState, prob, reward in mdp.succProbReward(state, action))
                  
    iterations = 0
    while True:
        iterations+=1
        newV={}
        for state in mdp.states():

            if mdp.isEnd(state):
                    newV[state] = 0
            else:

                newV[state] = max(Q(state,action) for action in mdp.actions(state))
                newV[state] = -float('inf')
                for action in mdp.actions(state):
                    temp = 0
                    for nextState, prob, reward in mdp.succProbReward(state, action):
                        temp +=  prob*(reward + V[nextState])
                    if newV[state] < temp:
                        newV[state] = temp
        if V == newV:
            break
        V = newV 

    print(iterations)
    return V   




mdp = TransportationMDP(N=10)
print(valueIteration(mdp))
