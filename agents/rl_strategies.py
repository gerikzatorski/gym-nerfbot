#!/usr/bin/env python

# rl_strategies.py (v0.0.1)
# Gerik Zatorski

import numpy as np
import os
import argparse 
import logging
import sys
import time

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
    def act(self, observation, reward, done):
        return self.action_space.sample()

class QLearning(Agent):
    """ QLearning Agent class """
    def act(self, observation, reward, done):
        raise NotImplementedError

###############################################################################
#                                     Main                                    #
###############################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Various RL algorithms for use with OpenAI.')
    parser.add_argument('-p', '--policy', default='random', choices=['random','epsilon_greedy'], help='Policy / Strategy to use. (Default: random)')
    parser.add_argument('-e', '--environment', default='OneStaticCircleTarget-v0', help='Name of the OpenAI Gym environment. (Default: OneStaticCircleTarget-v0')
    parser.add_argument('-n', '--nepisode', default='5', type=int, help="Number of episode. (Default: 5000)")
    parser.add_argument('-v', '--video', default='True', type=bool, help="Option to record video with OpenAI. (Default: 100)")

    args = parser.parse_args()

    # Custom logger setup
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # alternative is logging.WARN

    # Setup environment
    env = gym.make(args.environment)
    outdir = '/tmp/{0}-agent-results'.format(str(args.environment).lower())
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    
    # Setup agent policy/strategy with switch
    if args.policy == 'random': agent = RandomAgent(env.action_space)
    else: raise NotImplementedError

    # RL loop variables
    max_episodes = args.nepisode
    record_video = args.video
    reward = 0
    done = False

    # RL loop
    for i in range(max_episodes):
        print "EPISODE {0}".format(i)
        ob = env.reset()
        while True:

            time.sleep(0.1)

            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(action)
            print "{0} {1} {2} {3}".format(ob, reward, done, info)

            if done:
                break
            # record video?
            if (record_video): env.render('human')
            
    # Close the env and write monitor result info to disk
    env.close()

    # Upload to the scoreboard
    # gym.upload(outdir)

    logger.info("Successfully ran Simulation. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
