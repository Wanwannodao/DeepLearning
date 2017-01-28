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
def set_sarray_view(dst, g_param):
    for name, param in dst.namedparams():
        if name in g_param:
            # np.frombuffer gives the view without copying the data
            # this means that we can use s_array as shaerd array,
            # and this conversion is the fastes way.
            param.data = np.frombuffer(
                g_param[name], dtype=param.data.dtype).reshape(param.data.shape)

def copy_params(dst, src):
    dst.copyparams(src)

def copy_grads(dst, src):
    dst.cleargrads()
    dst.addgrads(src)

def work(pid, state_dim, out_dim):
    # create local net
    network = copy.deepcopy(global_network)

    
    if (pid == 0):
        #print ("global grad       @%s")%hex(id(global_network.fc0.W.grad.__array_interface__['data'][0]))
        #print ("global grad       @%s")%hex(id(global_network.fc0.W.grad.__array_interface__['data'][0]))

        print ("[pid 0]deep copied array @%s")%hex(network.fc0.W.data.ravel().__array_interface__['data'][0])
        print ("[pid 0]shared array      @%s")%hex(global_network.fc0.W.data.ravel().__array_interface__['data'][0])
   
    # all processes share the shared array
    set_sarray_view(network, global_params)
    print ("[pid %d] Set pramas to global params") % pid
    
    network.cleargrads()
    out = network(np.asarray([[1]], dtype=np.float32))
    out.backward()
    copy_grads(global_network, network)
 
if __name__ == "__main__":

    # master network 
    global_network = DummyNet(1, 1)

    # store global params to shared array
    # {name, rawarray}
    global_params = {}
    for name, param in global_network.namedparams():
        # param.data is numpy ndarray
        # param.data.ravel() is a contiguous flattend array
        global_params[name] = mp.RawArray(ctypes.c_int, 
                                          param.data.ravel())

    print ("original global param @%s")%hex(param.data.ravel().__array_interface__['data'][0])
    print ("raw array             @%s")%hex(global_params[name]._wrapper.get_address())

    set_sarray_view(global_network, global_params)

    workers = []
    print ("======= start parallel ======")
    for pid in xrange(mp.cpu_count()):
        p = mp.Process(target=work, args=(pid, 1, 1))
        workers.append(p)
        p.start()

    [p.join() for p in workers]
