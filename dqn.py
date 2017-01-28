import copy
import random
import collections

import gym
import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers

import matplotlib.pyplot as plt

class QFunction(chainer.Chain):

    def __init__(self, n_actions):
        initializer = chainer.initializers.HeNormal()
        c1 = 32
        c2 = 64
        c3 = 64
        fc_unit = 256

        super(QFunction, self).__init__(
             # the size of the inputs to each layer will be inferred
            conv1=L.Convolution2D(4, c1, 8, stride=4, pad=0),
            conv2=L.Convolution2D(c1, c2, 4, stride=2, pad=0),
            conv3=L.Convolution2D(c2, c3, 3, stride=1, pad=0),
            #conv4=L.Convolution2D(64, c4, 3, stride=1, pad=1),
            fc1=L.Linear(3136, fc_unit, initialW=initializer),
            fc2=L.Linear(fc_unit, n_actions, initialW=initializer),
            #bnorm1=L.BatchNormalization(c1),
            #bnorm2=L.BatchNormalization(c2),
            #bnorm3=L.BatchNormalization(c3),
            #bnorm4=L.BatchNormalization(c4),
        )

    def __call__(self, x):
        x = x/255.
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        #h = F.max_pooling_2d(F.relu(self.bnorm4(self.conv4(h))), 2, stride=2)
        h = F.relu(self.fc1(h))
        y = self.fc2(h)        
        return y

def get_greedy_action(Q, obs):
    xp = Q.xp
    obs = xp.expand_dims(xp.asarray(obs, dtype=np.float32), 0)
    with chainer.no_backprop_mode():
        q = Q(obs).data[0]
    return int(xp.argmax(q))

def mean_clipped_loss(y, t):
    # Add an axis because F.huber_loss only accepts arrays with ndim >= 2
    y = F.expand_dims(y, axis=-1)
    t = F.expand_dims(t, axis=-1)
    return F.sum(F.huber_loss(y, t, 1.0)) / y.shape[0]

def update(Q, target_Q, opt, samples, gamma=0.99, target_type='double_dqn'): 
    xp = Q.xp
    s = np.ndarray(shape=(minibatch_size, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), dtype=np.float32)
    a = np.asarray([sample[1] for sample in samples], dtype=np.int32)
    r = np.asarray([sample[2] for sample in samples], dtype=np.float32)
    done = np.asarray([sample[3] for sample in samples], dtype=np.float32)
    s_next = np.ndarray(shape=(minibatch_size, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), dtype=np.float32)

    for i in xrange(minibatch_size):
        s[i] = samples[i][0]
        s_next[i] = samples[i][4]

    # to gpu if available
    s = xp.asarray(s)
    a = xp.asarray(a)
    r = xp.asarray(r)
    done = xp.asarray(done)
    s_next = xp.asarray(s_next)
    
    # Prediction: Q(s,a)
    y = F.select_item(Q(s), a)
    
    f0 = Q.conv1.data
    print f0.shape
    # Target: r + gamma * max Q_b (s',b)
    with chainer.no_backprop_mode():
        if target_type == 'dqn':
            t = r + gamma * (1 - done) * F.max(target_Q(s_next), axis=1)
        elif target_type == 'double_dqn':
            t = r + gamma * (1 - done) * F.select_item(
                target_Q(s_next), F.argmax(Q(s_next), axis=1))
        else:
            raise ValueError('Unsupported target_type: {}'.format(target_type))
    loss = mean_clipped_loss(y, t)
    Q.cleargrads()
    loss.backward()
    opt.update()

def meanQvalue(Q, samples): 
    xp = Q.xp
    s = np.ndarray(shape=(minibatch_size, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), dtype=np.float32)
    a = np.asarray([sample[1] for sample in samples], dtype=np.int32)

    for i in xrange(minibatch_size):
        s[i] = samples[i][0]

    # to gpu if available
    s = xp.asarray(s)
    a = xp.asarray(a)

    # Prediction: Q(s,a)
    y = F.select_item(Q(s), a)
    mean_Q = (F.sum(y)/minibatch_size).data
    return mean_Q

STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

class ObsProcesser:
    def __init__(self):
        self.state = None
        
    def init_state(self, obs):
        processed_obs = self._preprocess_observation(obs)
        state = [processed_obs for _ in xrange(STATE_LENGTH)]
        self.state = np.stack(state, axis=0)
        
    def obs2state(self, obs):
        processed_obs = self._preprocess_observation(obs)
        self.state = np.concatenate((self.state[1:, :, :], processed_obs[np.newaxis]), axis=0)
        return self.state
    
    def _preprocess_observation(self, obs):
        # clop center
        return np.asarray(resize(rgb2gray(obs), (110, 84))[-84:, :]*255, dtype=np.uint8)

# Hyperparameters
# env_name = 'CartPole-v0'  # env to play
env_name = 'Breakout-v0'  # env to play
# env_name = 'Pong-v0'  # env to play
#env_name = 'MsPacman-v0'  # env to play
#env_name = 'SpaceInvaders-v0'  # env to play


M = 100000  # number of episodes
replay_start_size = 5000  # steps after which we start to update
steps_to_decay_epsilon = 1000000  # steps to take to decay epsilon
min_epsilon = 0.1  # minimum value of epsilon
sync_interval = 10000  # interval of target sync
evaluation_interval = 50000
update_inverval = 4
minibatch_size = 64  # size of minibatch
reward_scale = 1  # scale factor for rewards
gpu = 0  # gpu id (-1 to use cpu)
render = False  # open a rendering window (will not work without display)
target_type = 'dqn'  # 'dqn' or 'double_dqn'

# Initialize an environment
env = gym.make(env_name)
ndim_obs = env.observation_space.low.size
n_actions = env.action_space.n

# Initialize variables
D = collections.deque(maxlen=10 ** 6)  # replay memory
Rs = []  # past returns
step = 0  # total steps taken

# 
obs_processer = ObsProcesser()

# https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_atari.py
# We patch the environment to be closer to what Mnih et al. actually do: The environment
# repeats the action 4 times and a game is considered to be over during training as soon as a live
# is lost.
def _step(a):
    reward = 0.0
    action = env._action_set[a]
    lives_before = env.ale.lives()
    for _ in range(4):
        reward += env.ale.act(action)
    ob = env._get_obs()
    done = env.ale.game_over()
    return ob, reward, done, {}
env._step = _step

# Initialize chainer models
Q = QFunction(n_actions)
if gpu >= 0:
    chainer.cuda.get_device(gpu).use()
    Q.to_gpu(gpu)
target_Q = copy.deepcopy(Q)
opt = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
opt.setup(Q)


render = True  # open a rendering window (will not work without display)


fig, ax = plt.subplots(1,1)
plt.pause(.01)

for episode in range(M):


    obs = env.reset()

    obs_processer.init_state(obs)
    done = False
    R = 0.0
    t = 0
    limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    state = obs_processer.state
    
    while not done and t < limit:

        # Select an action
        epsilon = 1.0 if len(D) < replay_start_size else \
            max(min_epsilon, np.interp(
                step, [replay_start_size, replay_start_size+steps_to_decay_epsilon], [1.0, min_epsilon]))
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            a = get_greedy_action(Q, state)

        # Execute an action
        new_obs, r, done, _ = env.step(a)
            
        new_state = obs_processer.obs2state(new_obs)
        
        if render:
            env.render()
        R += r

        # Store a transition
        D.append((state, a, r * reward_scale, done, new_state))
        state = new_state

        # Sample a random minibatch of transitions
        if len(D) >= replay_start_size:
            if step % update_inverval == 0:
                samples = random.sample(D, minibatch_size)
                update(Q, target_Q, opt, samples, target_type=target_type)
            
            if step % sync_interval == 0:
                mean_Q = meanQvalue(Q, samples)
                print('target Q update! mean Q value : {}, epsilon:{}'.format(mean_Q, epsilon))
                
            #if step % evaluation_interval == 0:
            #    Evaluation(render_flag=False)

        if step % sync_interval == 0:
            target_Q = copy.deepcopy(Q)

        step += 1
        t += 1

    Rs.append(R)
    average_R = np.mean(Rs[-100:])
    if episode % 10 is 0:
        print('episode: {} step: {} R:{} average_R:{}'.format(
              episode, step, R, average_R))
    
    ax.clear()
    ax.plot(Rs)
    fig.canvas.draw()
