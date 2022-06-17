from email import policy
from pydoc import plain
from numpy import place


class HavingGame:
    def __init__(self, N) -> None:
        self.N = N
    
    def startState(self):
        return (+1, self.N)

    def isEndState(self, state):
        return state[1]==0

    def utility(self, state):
        player, number = state
        assert number == 0
        return player*float('inf')
     
    def actions(self, state):
        return ['-','/']

    def player(self, state):
        return state[0]

    def succ(self, state, action):
        player, number = state
        if action == '-':
            return (-player, number -1)
        elif action =='/':
            return (-player, number//2)


def humanPolicy(game, state):
    action = input('Input action:')
    if action in game.actions(state):
        return action


def minmaxPolicy(game: HavingGame, state):
    
    def recurse(state):
        if game.isEndState(state):
            return (game.utility(state), 'none')
         
        choices = [(recurse(game.succ(state,action))[0], action) for action in game.actions(state)]
        
        if game.player(state) == +1:
            return max(choices)
        elif game.player(state)==-1:
            return min(choices)

    value, action = recurse(state)
    print('minmax says action ={}, value = {}'.format(action, value))
    return action


policies ={+1: humanPolicy, -1: minmaxPolicy}
game = HavingGame(N=15)
state = game.startState()

while not game.isEndState(state):
    print('='*10, state)
    player = game.player(state)
    policy = policies[player]
    action = policy(game, state)
    state = game.succ(state, action)

print('utility = {}'.format(game.utility(state)))