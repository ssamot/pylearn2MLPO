__author__ = 'ssamot'

from sknn.pylearn2mplo import pylearn2MLPO, IncrementalMinMaxScaler
import numpy as np

class DeepQ():



    """
    A Q learning agent
    """

    def __init__(self, layers, state_action_preproc,  dropout = False, input_scaler=IncrementalMinMaxScaler(), output_scaler=IncrementalMinMaxScaler(),   learning_rate=0.01, verbose=0):
        self.memory = []
        self.network = pylearn2MLPO(layers, dropout, input_scaler, output_scaler, learning_rate,verbose)
        ##self.target_network = pylearn2MLPO()
        self.target_network = self.network
        self.gamma = 0.9
        self.epsilon = 0.1
        self.state_action_preproc = state_action_preproc
        self.swap_iterations = 10000
        self.swap_counter = 0

        self.initialised = False



    def __maxQ(self, state):
        Q = np.array([self.target_network.predict(state_action.reshape(1,state_action.size) )for state_action in self.state_action_preproc.enum(state)])
        #print Q
        return Q.max()


    def fit(self,state, next_state, action, reward, terminal):
        gamma = self.gamma

        maxQ = self.__maxQ(next_state)
        saf  = self.state_action_preproc.get(state, action)

        target = reward  + (1-terminal) * gamma * maxQ
        self.network.fit(saf.reshape(1,saf.size), np.array([[target]]))
        if(self.swap_counter % self.swap_iterations ==0 ):
            pass



    # e-greedy
    def act(self,state):

        if(not self.initialised):
            state_actions = list(self.state_action_preproc.enum(state))
            sa = state_actions[0].reshape(1,state_actions[0].size)
            target = np.array([[0]])
            #print target.shape, sa.shape
            self.network.fit(sa,target)

        b_action = np.array([self.target_network.predict(state_action.reshape(1,state_action.size) )for state_action in self.state_action_preproc.enum(state)]).argmax()



        if(np.random.random() < self.epsilon):
            r =  np.random.randint( 0,len(state_actions))
            #print "returning random action", r
            return r
        else:
            #print "returning best action", b_action
            return b_action






