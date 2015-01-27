#!/usr/bin/env python
__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

""" This example demonstrates how to use the discrete Temporal Difference
Reinforcement Learning algorithms (SARSA, Q, Q(lambda)) in a classical
fully observable MDP maze task. The goal point is the top right free
field. """

from scipy import *
from deepq.deepq import DeepQ
import numpy as np

from pybrain.rl.environments.mazes import Maze, MDPMazeTask


class MazePreProc():
    def __init__(self):
        self.actions = [0,1,2,3]

    def get(self, state, action):
        saf_onehot = np.zeros(88)
        saf_onehot[state[0]] = 1
        saf_onehot[action + 84] = 1
        return saf_onehot
        #return (np.concatenate([state, [action]]))

    def enum(self,state):
        for action in self.actions:
            yield self.get(state,action)

# create the maze with walls (1)
envmatrix = array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 0, 0, 1, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 0, 0, 1, 0, 0, 0, 0, 1],
                   [1, 0, 0, 1, 0, 1, 0, 0, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1, 1, 0, 0, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1]])

env = Maze(envmatrix, (4, 4))




# create task
task = MDPMazeTask(env )

layers  =  [("RectifiedLinear", 110),("RectifiedLinear", 110), ("Linear", )]

deepQ = DeepQ(layers,MazePreProc(), learning_rate=0.01)

oldState = task.getObservation()

avg_reward  = 0
for i in range(0,10000000):


    action = deepQ.act(oldState)
    task.performAction([action])
    state = task.getObservation()

    reward = task.getReward()
    terminal = 0

    deepQ.fit(oldState,state,action,reward, terminal )

    oldState = state


    avg_reward+=reward
    if(i % 5000 == 0 ):
        print avg_reward/1000.0
        avg_reward = 0
        env.reset()





# create value table and initialize with ones

# standard exploration is e-greedy, but a different type can be chosen as well
# learner.explorer = BoltzmannExplorer()


