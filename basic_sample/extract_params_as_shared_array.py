import multiprocessing as mp
import numpy as np
import chainer
from chainer import links as L

# meaningless dummy network
class DummyNet(chainer.Chain):
    def __init__(self, n_in, n_out):
        super(DummyNet, self).__init__(
            fc0=L.Linear(n_in, n_out)
            )
        
    def __call__(self, x):
        return self.fc0(x)

if __name__ == "__main__":

    # master network 
    global_network = DummyNet(1, 1)
    global_params = {}
    for name, param in  global_network.namedparams():
        print ("[%s]")%name
        # param.data is numpy ndarray
        # param.data.ravel() is a contiguous flattend array
        print param.data.ravel()
        global_params[name] = mp.RawArray('f', 
                                          param.data.ravel()
                                          )
