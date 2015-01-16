__author__ = 'ssamot'

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

from sknn.pylearn2mplo import pylearn2MLPO
from sklearn.metrics import r2_score

import random

dataset = datasets.load_linnerud()
X = dataset["data"]
y = dataset["target"]


def test_multivariate_regression():
    layers = [("RectifiedLinear", 250), ("RectifiedLinear", 250), ("Linear", )]
    #layers = [("Maxout", 250, 2), ("Maxout", 250,2), ("Linear", )]
    #layers = [("Tanh", 250, 2), ("Tanh", 250,2), ("Linear", )]

    nn = pylearn2MLPO(layers, learning_rate=0.1, verbose = 10)

    mm = MinMaxScaler()
    X_new = mm.fit_transform(X)
    y_new = mm.fit_transform(y)

    for k in range(0, 10000):
        errors = []
        tf = list(range(0, X_new.shape[0]))
        random.shuffle(tf)
        for i in tf:
            #ds = data(X[i:i+1], targets[i:i+1])
            X_i = X_new[i:i + 1]
            y_i = y_new[i:i + 1]
            nn.fit(X_i, y_i)
            #print X_i, y_i
            # ann.monitor.report_epoch()
            # ann.monitor()

    y_pred = nn.predict(X_new)
    r_0 = r2_score(y_new[:,0],y_pred[:,0] )
    r_1 = r2_score(y_new[:,1],y_pred[:,1] )
    r_2 = r2_score(y_new[:,2],y_pred[:,2] )

    #print y_new
    #print y_pred
    assert(r_0 > 0.99)
    assert(r_1 > 0.99)
    assert(r_2 > 0.99)
    #assert(False)
