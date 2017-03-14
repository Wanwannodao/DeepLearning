#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
import cv2
import gc
from ddqn_pr import DDQN_PR

import os
import numpy as np

flags = tf.app.flags

# hyperparams
flags.DEFINE_integer("batch_size", 32, "size of state")
flags.DEFINE_integer("state_length", 4, "length of state")
flags.DEFINE_integer("replay_start_size", 10000, "replay start size")
flags.DEFINE_integer("decay", 500000, "eps decay")
flags.DEFINE_float("min_epsilon", 0.1, "minimum epsilon")
flags.DEFINE_integer("sync_freq", 500, "frequence of target nets update")
flags.DEFINE_float("reward_scale", 1e-2, "reward scale")
flags.DEFINE_integer("episode", 500, "episode length")
flags.DEFINE_integer("N", 50000, "size of PR")
flags.DEFINE_float("alpha", 0.5, "alpha")
flags.DEFINE_float("beta", 0.5, "beta")
flags.DEFINE_float("beta_decay", 5000000, "beta decay")
flags.DEFINE_float("replay_period", 1, "replay_period")
flags.DEFINE_integer("eval", 10, "eval freq")
# config
flags.DEFINE_bool("render", True, "render")
flags.DEFINE_bool("save", True, "True: save learned model")
flags.DEFINE_integer("save_freq", 1, "freq of saving params")
flags.DEFINE_bool("restore", True, "True: restore model")
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
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # for breakout
        #gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[34:210]
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
        last = self.state[:, :, 1:]
        self.state = np.concatenate( (last, pre[:,:,np.newaxis]), axis=2 )        
        del pre, last
        return self.state
    
def main(_):
    #env = gym.make("Frostbite-v0")
    env = gym.make ("MsPacman-v0")
    n_s = env.observation_space.shape[0]
    n_a = env.action_space.n

    pre = Preprocessor()

    with tf.Session() as sess:
        dqn = DDQN_PR(input_shape=[FLAGS.batch_size, 84, 84, FLAGS.state_length], action_n=n_a, N=FLAGS.N,
                  alpha=FLAGS.alpha, beta=FLAGS.beta, beta_decay=FLAGS.beta_decay)
        global_step = 0

        saver = tf.train.Saver()
        if FLAGS.restore and os.path.exists("./checkpoint/model.ckpt"):
            saver.restore(sess, "./checkpoint/model.ckpt")
        else:
            sess.run(tf.global_variables_initializer())
        for episode in range(FLAGS.episode):
            obs = env.reset()
            #s = env.reset()
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
                if global_step < FLAGS.replay_start_size or np.random.rand() < epsilon:
                    a = env.action_space.sample()
                else:
                    a = dqn.greedy(s[np.newaxis], sess)

                obs, r, done, _ = env.step(a)
                s_ = pre.get_state(obs)
                #s_, r, done, _ = env.step(a)
                
                if FLAGS.render:
                    env.render()

                if global_step < FLAGS.replay_start_size - 1:
                    dqn.set_exp((s, a, r*FLAGS.reward_scale, done, s_),
                                init=True)
                elif global_step == FLAGS.replay_start_size - 1:
                    dqn.pr.rebalance()
                    dqn.set_exp((s, a, r*FLAGS.reward_scale, done, s_))
                else:
                    dqn.set_exp((s, a, r*FLAGS.reward_scale, done, s_))
                                        

                if global_step % FLAGS.N == 0:
                    dqn.pr.rebalance()

                R += r                
                s = s_
                
                if global_step >= FLAGS.replay_start_size and global_step % FLAGS.replay_period == 0:
                    dqn.pr._interval()
                    dqn.update(sess)

                if global_step % FLAGS.sync_freq == 0:
                    dqn.update_target(sess)
                
                step += 1
                global_step += 1
              
            if FLAGS.save and episode % FLAGS.save_freq == 0:
                saver.save(sess, "./checkpoint/model.ckpt")

            # Evaluation
            if episode % FLAGS.eval == 0:
                obs = env.reset()
                pre.init(obs)
                done = False
                s = pre.state
                R = 0
                step = 0
                epsilon = 0.01
                while not done and step < limit:
                    if np.random.rand() < epsilon:
                        a = env.action_space.sample()
                    else:
                        a = dqn.greedy(s[np.newaxis], sess)

                    obs, r, done, _ = env.step(a)
                    s_ = pre.get_state(obs)
                    s = s_
                    if FLAGS.render:
                        env.render()

                    R += r
                    step += 1
                    
                print("epoch:{}, step:{}, R:{}".format(episode, global_step, R))
                with open('R.csv', 'a')  as f:
                    
                    f.write("{},".format(R))
                gc.collect()                
            
if __name__ == "__main__":
    tf.app.run()
