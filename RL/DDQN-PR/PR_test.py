from pr import PrioritizedReplayBuf

pr = PrioritizedReplayBuf(10000, 0.5, 1.0, 10000, 32)

for i in range(20000):
    pr.insert((i,1, 1, 1, 1))
"""
print("{}".format(pr.D))
print("{}".format(pr.heap))

#a = (3,)
#pr._update(i=0, delta=0.5)
print("{}".format(pr.D))
print("{}".format(pr.heap))

pr.insert((4))
print("{}".format(pr.D))
print("{}".format(pr.heap))

pr.insert((5))
print("{}".format(pr.D))
print("{}".format(pr.heap))

pr.insert((6))
print("{}".format(pr.D))
print("{}".format(pr.heap))

pr.insert((7))
print("{}".format(pr.D))
print("{}".format(pr.heap))
"""

import numpy as np
import matplotlib.pyplot as plt

#dist, cdf = pr._interval()
#y = cdf[pr.seg[:-1]]


#fig = plt.figure()
#ax = fig.add_subplot(211)
#ax2 = fig.add_subplot(212)

#ax.clear()
#ax.plot(dist)
#ax.set_xlabel("i")
#ax.set_ylabel("P(i)")
#ax.grid(True)
#ax2.clear()
#ax2.plot(cdf)
#ax2.set_xlabel("i")
#ax2.set_ylabel("$P(I <= i)$")
#ax2.grid(True)

#ax2.vlines(pr.seg, 0, 1, linestyle='dotted')
#ax3.plot(pr.seg[:-1], y, 'o')



#ax.set_vline([0, 100], 0.009, 0.015, linestyles="dashed")
#fig.savefig('dist.png')


#pr.interval()
#print("{}".format(pr.seg))

#h, d, is_w = pr.stratified_sample()



#print("{}".format(rank))
#is_w = ((1.0/d[0]['heap'])**pr.alpha * pr.p_n) ** pr.beta
#print("{}".format(is_w))

