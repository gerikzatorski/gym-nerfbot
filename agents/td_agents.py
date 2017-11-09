#!/usr/bin/env python

# rl_strategies.py (v0.0.1)
# Gerik Zatorski

import numpy as np
import os
import argparse 
import logging
import sys
import time
import random

import gym
from gym import wrappers
import gym_nerfbot

class Agent(object):
    """ Policy/Strategy interface for agent classes. """
    def __init__(self, action_space):
        self.action_space = action_space
        # could also call super(RandomAgent, self).__init__() from subclasses

###############################################################################
#                                   Policies                                  #
###############################################################################

class RandomAgent(Agent):
    """ Agent with random policy """
    def act(self, state):
        return self.action_space.sample()

class QLearner(Agent):
    """ QLearner Agent class (epsilon greedy)"""
    def __init__(self, env, epsilon=0.2, alpha=0.5, gamma=0.8):
        # state action value function
        self.q = np.zeros([ env.observation_space.n , env.action_space.n ])

        # Q-learning params
        self.epsilon = epsilon  # epsilon greedy
        self.alpha = alpha      # learning rate (0-1)
        self.gamma = gamma      # discount factor (0-1) - converges slower at 1

        self.action_space = env.action_space
        self.observation_space = env.observation_space 

    def update(self, s, a, reward, ns):
        """ TODO: simplify this procedure to generalize to other algorithms"""
        self.q[s,a] = reward + np.max(self.q[ns,:])
    
    def act(self, state):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else: # else explore greedily
            vector = self.q[state,:]
            m = np.amax(vector)
            indices = np.nonzero(vector == m)[0]
            return random.choice(indices)

    ###############################################################################
#                                     MAIN                                    #
###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Various algorithms for OpenAI.')
    parser.add_argument('-p', '--policy', default='random', choices=['random','qlearner'], help='Policy / Strategy to use. (Default: random)')
    parser.add_argument('-e', '--environment', default='OneStaticCircleTarget-v0', help='Name of the OpenAI Gym environment. (Default: OneStaticCircleTarget-v0')
    parser.add_argument('-n', '--nepisode', default='5', type=int, help="Number of episode. (Default: 5)")
    parser.add_argument('-s', '--maxsteps', default='100', type=int, help="Max steps per episode. (Default: 100)")
    parser.add_argument('-v', '--video', default='True', type=bool, help="Option to record video with OpenAI. (Default: True)")
    args = parser.parse_args()

    # Setup environment
    env = gym.make(args.environment)
    outdir = '/tmp/{0}-agent-results'.format(str(args.environment).lower())
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0) # TODO: rm?
    
    # Setup agent policy/strategy with switch statement
    if args.policy == 'random': agent = RandomAgent(env.action_space)
    elif args.policy == 'qlearner': agent = QLearner(env)
    else: raise NotImplementedError

    # RL loop variables
    max_episodes = args.nepisode
    max_steps = args.maxsteps
    record_video = args.video
    reward = 0
    done = False

    # RL loop
    for e in range(max_episodes):
        print "EPISODE {0}".format(e)
        state = env.reset()
        for i in range(max_steps):
            time.sleep(0.1)
            action = agent.act(state)
            new_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, new_state)
            state = new_state

            print "{0} {1} {2} {3}".format(state, reward, done, info)
            if (record_video): env.render('human')
            if done: break
            
    env.close() # Close the env and write monitor result info to disk
    # gym.upload(outdir) # Upload to the scoreboard
    # logger.info("Successfully ran Simulation")
