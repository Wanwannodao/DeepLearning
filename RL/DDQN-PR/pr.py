import numpy as np
import math


class PrioritizedReplayBuf:
    
    def __init__(self, N, alpha, beta, beta_decay, batch_size):
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.beta_step = (1.0 - beta) / beta_decay
        self.k = batch_size
        self.k_ = 1.0/self.k
        self.p_n = None
        
        # Binary Heap
        self.heap = []
        # Underlying Repay Memory ( Circular Buffer )
        self.D = []
        self.cur = 0

        # segment for stratified sampling
        self.seg = []

    def _div(i, d):
        return d / (i + 1)

    def _interval(self):
        del self.seg
        self.seg = []

        n = self.N if len(self.D) == self.N else len(self.D)
        # distribution
        # maybe comprehension is faster than map
        #dist = np.asarray(list(map(lambda d: d[1]/(d[0]+1), enumerate(dist))), dtype=np.float32)
        dist = np.asarray([(1.0/(i+1))**self.alpha for i in range(n)], dtype=np.float32)
        self.p_n = np.sum(dist)
        dist = dist / self.p_n

        # for IS weights
        self.p_n = self.p_n / n

        # cumulative distibution
        cdf = np.cumsum(dist)

        unit = (1.0 - cdf[0])/self.k
        # comprehension is faster
        self.seg = [ np.searchsorted(cdf, cdf[0]+unit*i) for i in range(self.k) ]

        # sentinel
        self.seg.append(n)

        del dist

        # for debug
        # return dist, cdf

    def set_aplha(self, alpha):
        self.alpha = alpha
        self._interval()

    # linearly annealing
    def _anneal_beta(self):
        self.beta += self.beta_step

    def stratified_sample(self):
        
        h_indices =[ np.random.randint(self.seg[i], self.seg[i+1]) \
                   for i in range(self.k) ]
        h = np.asarray(self.heap)[h_indices]
        d_indices = [i['D'] for i in h]
        d = np.asarray(self.D)[d_indices]
        
        rank = np.asarray([ p['heap']+1 for p in d], dtype=np.int32)

        #print("seg:{} \n".format(self.seg)) 
        #print("rank:{} \n".format(rank))
        #for v in h:
        #    print ("{}, ".format(v['delta']))
        #print("\n")
        
        is_w = ((1.0 / rank**self.alpha) * self.p_n)** self.beta

        del h_indices, d_indices, rank
        return d, is_w

    def _f(self, i, h):
        self.D[h['D']]['heap'] = i
        
    def rebalance(self):
        self.heap = sorted(self.heap, key=lambda h: h['delta'], reverse=True)
        map(self._f, enumerate(self.heap)) 
        
        
    def _swapNode(self, i, p):
        # swap in D
        # indices in D
        p_d = self.heap[p]['D']
        x_d = self.heap[i]['D']
        self.D[ p_d ]['heap'], self.D[ x_d ]['heap'] \
            = self.D[ x_d ]['heap'], self.D[ p_d ]['heap']

        # swap in heap
        self.heap[p], self.heap[i] = self.heap[i], self.heap[p]
        
    def _replace(self, x):
        # replace oldest entory with new transition
        s, a, r, done, _ = self.D[self.cur]['transition']
        del s, a, r, done
        self.D[self.cur]['transition'] = x

        # corresponding ref in heap
        h = self.D[self.cur]['heap']

        # maximum delta = maximum priority
        self._update(h, delta=1.0)


        self.cur += 1
        self.cur %= self.N
        
    
    def _update(self, i, delta):
        
        self.heap[i]['delta'] = delta
        
        p = math.ceil(i/2) - 1 # parent indice
        if p > 0 and delta > self.heap[p]['delta']:
            while p > 0 and delta > self.heap[p]['delta']:

                self._swapNode(i, p)
                i = p
                p = math.ceil(i/2) - 1
        else:
            c = (i << 1) + 1 # child indice
            while c < len(self.heap):
                b = -1
                if c+1 < len(self.heap):
                    b = self.heap[c+1]['delta']

                if delta < self.heap[c]['delta'] or delta < b:
                    if self.heap[c]['delta'] < b:
                        self._swapNode(i, c+1)
                        i = c+1
                    else:
                        self._swapNode(i, c)
                        i = c
                    c = (i << 1) + 1 
                else:
                    break
    
    def insert(self, x, init=False):
        if len(self.D) < self.N:
            i = len(self.D)
            self.D.append({'transition': x, 'heap': i})
            self.heap.append({'delta': 1.0, 'D': i})
            
            if not init:
                self._update(i, delta=1.0)
        else:
            self._replace(x)
        
    def update_delta(self, d, delta):
        for e, v in zip(d, delta):
            self._update(e['heap'], v)

        self._anneal_beta()
        
        
