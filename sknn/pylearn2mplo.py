import theano
import numpy as np
import time
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.training_algorithms import sgd
import scipy.io
import random
from pylearn2.models import mlp, maxout
from pylearn2.costs.mlp.dropout import Dropout



class pylearn2MLPO():

    """
    SK-learn like interface for pylearn2
    Notice how training the model and the training algorithm are now part of the same class, which I actually quite like
    This class is focused a bit on online learning, so you might need to modify it to include other pylearn2 options if
    you have access all your data upfront
    """

    def __init__(self, layers, dropout = False, input_normaliser = None, output_normaliser = None,   learning_rate=0.01, verbose=0):
        """

        :param layers: List of tuples of types of layers alongside the number of neurons
        :param learning_rate: The learning rate for all layers
        :param verbose: Verbosity level
        :return:
        """
        self.layers = layers
        self.ds = None
        self.f = None
        self.verbose = verbose
        cost = None
        if(dropout):
            cost = Dropout()
        self.trainer = sgd.SGD(learning_rate=learning_rate, cost = cost, batch_size=1)

    def __linit(self, X, y):
        if(self.verbose > 0):
            print "Lazy initialisation"

        layers = self.layers
        pylearn2mlp_layers = []
        self.units_per_layer = []
        #input layer units
        self.units_per_layer+=[X.shape[1]]

        for layer in layers[:-1]:
            self.units_per_layer+=[layer[1]]

        #Output layer units
        self.units_per_layer+=[y.shape[1]]

        if(self.verbose > 0):
            print "Units per layer", str(self.units_per_layer)


        for i, layer in enumerate(layers[:-1]):

            fan_in = self.units_per_layer[i] + 1
            fan_out = self.units_per_layer[i+1]
            lim = np.sqrt(6) / (np.sqrt(fan_in + fan_out))
            layer_name = "Hidden_%i_%s"%(i,layer[0])
            activate_type = layer[0]
            if activate_type == "RectifiedLinear":
                hidden_layer = mlp.RectifiedLinear(
                    dim=layer[1],
                    layer_name=layer_name,
                    irange=lim)
            elif activate_type == "Sigmoid":
                hidden_layer = mlp.Sigmoid(
                    dim=layer[1],
                    layer_name=layer_name,
                    irange=lim)
            elif activate_type == "Tanh":
                hidden_layer = mlp.Tanh(
                    dim=layer[1],
                    layer_name=layer_name,
                    irange=lim)
            elif activate_type == "Maxout":
                hidden_layer = maxout.Maxout(
                    num_units=layer[1],
                    num_pieces=layer[2],
                    layer_name=layer_name,
                    irange=lim)
            else:
                raise NotImplementedError(
                    "Layer of type %s are not implemented yet" %
                    layer[0])
            pylearn2mlp_layers += [hidden_layer]

        output_layer_info = layers[-1]
        output_layer_name = "Output_%s"%output_layer_info[0]

        fan_in = self.units_per_layer[-2] + 1
        fan_out = self.units_per_layer[-1]
        lim = np.sqrt(6) / (np.sqrt(fan_in + fan_out))

        if(output_layer_info[0] == "Linear"):
            output_layer = mlp.Linear(
                dim=self.units_per_layer[-1],
                layer_name=output_layer_name,
                irange=lim)
            pylearn2mlp_layers += [output_layer]

        self.mlp = mlp.MLP(pylearn2mlp_layers, nvis=self.units_per_layer[0])
        self.ds = DenseDesignMatrix(X=X, y=y)
        self.trainer.setup(self.mlp, self.ds)
        inputs = self.mlp.get_input_space().make_theano_batch()
        self.f = theano.function([inputs], self.mlp.fprop(inputs))

    def fit(self, X, y):
        """
        :param X: Training data
        :param y:
        :return:
        """
        if(self.ds is None):
            self.__linit(X, y)

        ds = self.ds
        ds.X = X
        ds.y = y
        self.trainer.train(dataset=ds)
        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """
        return self.f(X)

