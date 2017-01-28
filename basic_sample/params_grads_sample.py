import multiprocessing as mp
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


def copy_params(dst, src):
    dst.copyparams(src)

def copy_grads(dst, src):
    dst.cleargrads()
    dst.addgrads(src)

def work(pid, state_dim, out_dim):
    # create local net
    network = DummyNet(state_dim, out_dim)

    copy_params(network, global_network)
    print ("[pid %d] Set pramas to global params") % pid
    
    network.cleargrads()
    out = network(np.asarray([[1]], dtype=np.float32))
    out.backward()

    copy_grads(global_network, network)
    print ("[pid %d] Set global grads to grads") % pid
if __name__ == "__main__":

    # master network 
    global_network = DummyNet(1, 1)

    workers = []
    for pid in xrange(mp.cpu_count()):
        p = mp.Process(target=work, args=(pid, 1, 1,))
        workers.append(p)
        p.start()

    [p.join() for p in workers]
    

