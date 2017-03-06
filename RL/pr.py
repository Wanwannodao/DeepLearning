import numpy as np
import math


class PrioritizedReplayBuf:
    
    def __init__(self, N, alpha, beta, batch_size):
        self.N = N
        self.alpha = alpha
        self.beta = beta
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

    def _interval(self):
        # distribution
        dist = np.asarray([1.0/(i+1) for i in range(self.N)], dtype=np.float32)
        dist = dist**self.alpha
        self.p_n = np.sum(dist)
        dist = dist / self.p_n
        # for IS weights
        self.p_n = 1.0 / (self.p_n*self.N)

        # cumulative distibution
        cdf = np.cumsum(dist)
        for i in range(self.N):
            if cdf[i] > len(self.seg)*self.k_:
                self.seg.append(i)
        self.seg.append(self.N)

    def set_aplha(self, alpha):
        self.alpha = alpha
        self._interval()

    def annealing_beta(self, beta):
        self.beta = beta

    def stratified_sample(self):
        
        indices =[ np.random.randint(self.seg[i], self.seg[i+1]) \
                   for i in range(self.k) ]
        h = np.asarray(self.heap)[indices]
        indices = [i['D'] for i in h]
        d = np.asarray(self.D)[indices]
        
        rank = np.asarray([p['heap'] for p in d], dtype=np.float32)
        is_w = (rank**self.alpha * self.p_n) ** self.beta

        return h, d, is_w
        
        
    def _upHeap(self, i, p):
        """
        arg: i: x's indice on heap, p: parent indice
        return: upped x's indice
        """
               
        # swap in D
        # indices in D
        p_d = self.heap[p]['D']
        x_d = self.heap[i]['D']
        self.D[ p_d ]['heap'], self.D[ x_d ]['heap'] \
            = self.D[ x_d ]['heap'], self.D[ p_d ]['heap']

        # swap in heap
        self.heap[p], self.heap[i] = self.heap[i], self.heap[p]

    def _downHeap(self, i, c):
        
        c_d = self.heap[c]['D']
        x_d = self.heap[i]['D']
        self.D[ c_d ]['heap'], self.D[ x_d ]['heap'] \
            = self.D[ x_d ]['heap'], self.D[ c_d ]['heap']

        self.heap[c], self.heap[i] = self.heap[i], self.heap[c]
        

    def _replace(self, x):
        # replace oldest entory with new transition
        self.D[self.cur]['transition'] = x

        # corresponding indice in heap
        h = self.D[self.cur]['heap']

        # maximum delta = maximum priority
        self._update(h, delta=1.0)

        self.cur += 1        

    def _update(self, i, delta):
        self.heap[i]['delta'] = delta

        p = math.ceil(i/2) - 1 # parent indice

        while p > 0 and delta > self.heap[p]['delta']:

            self._upHeap(i, p)
            i = p
            p = math.ceil(i) - 1
            
        c = (i << 1) + 1 # child indice
        while c < len(self.heap):
            b = -1
            if c+1 < len(self.heap):
                b = self.heap[c+1]['delta']

            if delta < self.heap[c]['delta'] or delta < b:
                if self.heap[c]['delta'] < b:
                    self._downHeap(i, c+1)
                    i = c+1
                else:
                    self._downHeap(i, c)
                    i = c
                c = (i << 1) + 1 
            else:
                break
                    
    def insert(self, x):
        """
        arg: x: new transition, always has maximum priority
        """
        if len(self.D) < self.N:
            i = len(self.D)
            self.D.append({'transition': x, 'heap': i})
            self.heap.append({'delta': 1.0, 'D': i})
            self._update(i, delta=1.0)
        else:
            self._replace(x)
        
