import copy
import random
import collections
import gym
import numpy as py
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
fig2 = plt.figure(figsize=(7,7))
plt.pause(.01)

class FeatureExtractor(chainer.Chain):
    def __init__(self, n_actions):
        c1 = 32
        c2 = 64
        c3 = 64

        super(FeatureExtractor, self).__init__(
            conv1=L.Convolution2D(4, c1, 8, stride=4, pad=0),
            conv2=L.Convolution2D(c1, c2, 4, stride=2, pad=0),
            conv3=L.Convolution2D(c2, c3, 3, stride=1, pad=0)
            )

    def __call__(self, x):
        x = x/255.
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(x))
        return self.conv3(h)

class DDPG(chainer.Chain):
    def __init__(self, n_actions):
        initializer=chainer.initializers.HeNormal()
        fc_unit=256

        super(DDPG, self).__init__(
            fe=FeatureExtractor(n_actions),
            fc1=L.Linear(None, fc_unit, initialW=initializer),
            fc2=L.Linear(fc_unit, n_actions, initialW=initializer)
