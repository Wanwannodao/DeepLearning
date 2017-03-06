import multiprocessing as mp
import ctypes
import copy
import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L

# This sample spread params of master network to workers
# and set gradients of master network to those of workers

# meaningless dummy network
class DummyNet(chainer.Chain):
    def __init__(self, n_in, n_out):
        super(DummyNet, self).__init__(
            fc0=L.Linear(n_in, n_out)
            )
        
    def __call__(self, x):
        return self.fc0(x)

# set params to numpy array pointing to shared raw array
def set_sarray_view(dst_net, g_param, dst_opt, g_state):
    for name, param in dst_net.namedparams():
        if name in g_param:
            # np.frombuffer gives the view without copying the data
            # this means that we can use s_array as shaerd array,
            # and this conversion is the fastes way.
            param.data = np.frombuffer(
                g_param[name], dtype=param.data.dtype).reshape(param.data.shape)

    for name, state in dst_opt._states.items():
        if name in g_state:
            state.data = np.frombuffer(
                g_state[name], dtype=state.data.dtype).reshape(state.data.shape)

def copy_params(dst, src):
    dst.copyparams(src)

def copy_grads(dst, src):
    dst.cleargrads()
    dst.addgrads(src)

class Agent():
    def __init__(self, pid):
        self.pid = pid
        # create local net
        self.network = copy.deepcopy(global_network)
        print ("[pid %d] Set pramas to global params") % pid
                
    
    def run(self):
        self.network.cleargrads()
        out = self.network(np.asarray([[1]], dtype=np.float32))
        out.backward()
        copy_grads(global_network, self.network)
        
        copy_params(self.network, global_network)

def run(pid, ):
    agent = Agent(pid)
    agent.run()

 
if __name__ == "__main__":

    # master network 
    global_network = DummyNet(1, 1)

    # store global params to shared array
    # {name, rawarray}
    global_params = {}
    for name, param in global_network.namedparams():
        # param.data is numpy ndarray
        # param.data.ravel() is a contiguous flattend array
        global_params[name] = mp.RawArray(ctypes.c_float, 
                                          param.data.ravel())
    global_opt = chainer.optimizers.Adam()
    global_opt.setup(global_network)
    global_opt_states = {}
    for name, state in global_opt._states.items():
        print state
        #global_opt_params[name] = mp.RawArray(ctypes.c_float, # TODO is this int??
        #                                      state.ravel())

    set_sarray_view(global_network, global_params, 
                    global_opt, global_opt_states)

    workers = []
    for pid in xrange(mp.cpu_count()):
        p = mp.Process(target=run, args=(pid,))
        workers.append(p)
        p.start()

    [p.join() for p in workers]
