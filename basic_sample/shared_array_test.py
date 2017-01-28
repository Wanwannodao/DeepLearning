import multiprocessing as mp

import numpy as np
import ctypes

def work(pid):

    if pid==5:
        s_array[0] = 5
        array[0] = 5

# rawarray to nparray via np.asarray
def raw2np(raw):
    class Empty: pass
    array = Empty()
    array.__array_interface__ = {
        'data': (raw._wrapper.get_address(), False),
        'typestr': 'i',
        'descr': None,
        'shape': (raw._wrapper.get_size(),),
        'strides': None,
        'version': 3
    }
    return np.asarray(array).view(dtype=np.int32)
    

if __name__ == "__main__":

    raw_array= mp.RawArray(ctypes.c_int, 
                           np.asarray([1], dtype=np.int32)
                           )

    # np.frombuffer gives the view without copying the data
    # this means that we can use s_array as shaerd array,
    # and this conversion is the fastes way.
    s_array = np.frombuffer(raw_array, dtype=np.int32)

    # alternative way
    array = raw2np(raw_array)
    
    print s_array.__array_interface__['data'][0]
    print raw_array._wrapper.get_address()

    print ("%d %d")%(s_array, array[0])
    workers = []
    for pid in xrange(mp.cpu_count()):
        p = mp.Process(target=work, args=(pid,))
        workers.append(p)
        p.start()

    [p.join() for p in workers]
    print ("%d %d")%(s_array, array[0])
