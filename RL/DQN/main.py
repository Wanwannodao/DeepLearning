#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
import cv2
from dqn import DQN

import numpy as np

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 64, "size of state")
flags.DEFINE_integer("state_length", 4, "length of state")
flags.DEFINE_bool("render", True, "render")
flags.DEFINE_integer("replay_start_size", 64, "replay start size")
flags.DEFINE_integer("decay", 500000, "eps decay")
flags.DEFINE_float("min_epsilon", 0.01, "minimum epsilon")
flags.DEFINE_integer("sync_freq", 200, "frequence of target nets update")
flags.DEFINE_float("reward_scale", 1e-2, "reward scale")
flags.DEFINE_integer("episode", 500, "episode length")
FLAGS = flags.FLAGS
                        
# ===================
# Preprocessor
# ===================
class Preprocessor:
    def __init__(self):
        self.state = None
        
    def init(self, obs):
        pred = self._preprocess(obs)
        self.last = pred
        state = [pred for _ in range(FLAGS.state_length)]
        # [84, 84, 4]
        self.state = np.stack(state, axis=2)        

    def _preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[34:210]
        gray = cv2.resize(gray, (84, 84))
        """
        cv2.namedWindow("window")
        cv2.imshow("window", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        return (gray - 127.5) / 127.5

    def get_state(self, obs):
        pre = self._preprocess(obs)
        # removing flickering
        state = np.maximum(pre, self.last)
        self.last = pre
        
        self.state = np.concatenate( (self.state[:, :, 1:], state[:,:,np.newaxis]), axis=2 )
        return self.state
    
def main(_):
    env = gym.make("Breakout-v0")

    pre = Preprocessor()

    with tf.Session() as sess:
        dqn = DQN(input_shape=[FLAGS.batch_size, 84, 84, 4], action_n=4)
        global_step = 0
        
        sess.run(tf.global_variables_initializer())
        for episode in range(FLAGS.episode):
            obs = env.reset()
            pre.init(obs)
            done = False
            R = 0.0
            step = 0
            limit = env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps")
            s = pre.state
            
            while not done and step < limit:

                # epsilon decay
                epsilon = 1.0 if global_step < FLAGS.replay_start_size else \
                          max(FLAGS.min_epsilon, np.interp(
                              global_step, [0, FLAGS.decay], [1.0, FLAGS.min_epsilon]))
                
                # epsilon greedy
                if np.random.rand() < epsilon:
                    a = env.action_space.sample()
                else:
                    a = dqn.greedy(s, sess)

                obs, r, done, _ = env.step(a)
                s_ = pre.get_state(obs)

                if FLAGS.render:
                    env.render()

                #R += r
                              
                dqn.set_exp((s, a, r*FLAGS.reward_scale, done, s_))

                s = s_
                
                if global_step >= FLAGS.replay_start_size:
                    dqn.update(sess)

                if global_step % FLAGS.sync_freq:
                    dqn.update_target(sess)
                
                step += 1
                global_step += 1
            
if __name__ == "__main__":
    tf.app.run()


        
