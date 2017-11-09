#!/usr/bin/env python

# nerfbot_env.py
# Gerik Zatorski

import gym
from gym import spaces
from gym.utils import seeding
import gym_nerfbot

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
# import matplotlib as mpl
# mpl.rcParams['toolbar'] = 'None'

OBSERVATION_W = 500
OBSERVATION_H = 500
WINDOW_W = 500
WINDOW_H = 500
VIDEO_W = 500
VIDEO_H = 500

IMAGE_DIM = (OBSERVATION_W, OBSERVATION_H)
DEG_TO_RAD = math.radians(1)

###############################################################################
#                         Nerfbot Environment Interface                       #
###############################################################################

class NerfbotEnv(gym.Env):
    """ An interface for different types of Nerfbot simulations """
    metadata = {'render.modes': ['human', 'rgb_array', 'observation_pixels']}

    def __init__(self, obs_type='OneDimCoord', distance=10):
        self.fig = None
        self._obs_type = obs_type
        self.distance = distance               # characterize distance from Nerfbot to the simulated wall
        self.motor_limits = (28, 28)           # limit stepper motor ranges for safety
        self.home = (self.motor_limits[0]/2 ,  # calibrate for center of observation space
                     self.motor_limits[1]/2)
        self.state = self.home                 # set state to calibrated position

        # create noisy background for sim
        if self._obs_type =='OneDimCoord':
            self.img = np.zeros((OBSERVATION_H, OBSERVATION_W))
        if self._obs_type =='image':
            self.img = np.random.random((OBSERVATION_H, OBSERVATION_W))
        
        # assumption: camera border is equivalent to motor aim at limits
        self.step_angle = math.radians(1.8) # 1.8 deg per step
        self.virtual_width = self.distance * math.tan(self.step_angle * self.motor_limits[0])
        self.virtual_height = self.distance * math.tan(self.step_angle * self.motor_limits[0])

        # define observation and action spaces
        self.action_space = spaces.Discrete(self.motor_limits[0] * self.motor_limits[1])
        if self._obs_type =='OneDimCoord':
            self.observation_space = spaces.Discrete(OBSERVATION_W * OBSERVATION_H)
        elif self._obs_type =='image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(OBSERVATION_W, OBSERVATION_H, 3))
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def _virtual_motor_pos(self):
        """ Simlate a shot's position based on Nerfbot's pan and tilt steps to a position on a virtual wall (meters?)
        TODO: should I check that it is within the limits?
        """
        horiz = self.virtual_width/2 + (self.distance * math.tan(self.step_angle * (self.state[0] - self.home[0])))
        vert = self.virtual_height/2 + (self.distance * math.tan(self.step_angle * (self.state[1] - self.home[1])))
        return horiz, vert

    def _reset(self):
        """ OpenAI gym API: reset
        Returns: initial observation (determined by subclass)
        """
        self.human_render = False

    def _step(self, action):
        """ OpenAI gym API: step
        Override in ALL subclasses
        """
        raise NotImplementedError

    def _render(self, mode='human', close=False):
        """ OpenAI gym API: render
        Based off OpenAI CarRacing-v0
        """
        if close:
            if self.fig is not None:
                plt.close()
                self.fig = None
            return

        if self.fig is None:
            dpi = 80
            figsize = OBSERVATION_W / float(dpi), OBSERVATION_H / float(dpi)
            self.fig = plt.figure(figsize=figsize)
            # self.fig.set_tight_layout({"pad": .0})
            
        print "STEP"

        ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())

        ax.set_frame_on(False)
        ax.set_axis_off()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)


        self.fig.add_axes(ax)

        # add the background img
        ax.imshow(self.img, interpolation='nearest')

        # add target
        target = patches.Circle(self.target.image_coord, self.target.radius, edgecolor='none', facecolor='red')
        ax.add_patch(target)

        # add aiming crosshairs
        x = self.state[0] * VIDEO_W / self.motor_limits[0]
        y = self.state[1] * VIDEO_W / self.motor_limits[1]
        aim = patches.Circle((x,y), self.target.radius/4, edgecolor='none', facecolor='red')
        ax.add_patch(aim)
        
        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        self.fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # self.fig.savefig("test.png", bbox_inches='tight', pad_inches=0)

        
        if mode == 'human':
            plt.show(block=False)
        if mode == 'rgb_array':
            return np.array(data)
        else:
            pass

        # target.remove()
        # aim.remove()


###############################################################################
#                             Nerfbot Environments                            #
###############################################################################

class OneStaticCircleTarget(NerfbotEnv):
    """ OpenAI environment with one target
    Target never moves, not even between episodes (see reset)
    Because the target is static, no update is need through observation
    """
    def __init__(self, shape='Circle'):
        super(OneStaticCircleTarget, self).__init__()
        self.target = CircleTarget()

    def _step(self, action):
        assert self.action_space.contains(action)
        done = False
        reward = 0.0

        self.state = OneToTwoDim(action, self.motor_limits)

        aim_horiz, aim_vert = self._virtual_motor_pos()

        target_horiz, target_vert = self.target.virtual_target_position(self)

        dx = target_horiz - aim_horiz
        dy = target_vert - aim_vert
        distance = math.sqrt(dx**2 + dy**2)
        reward = -distance

        done = bool(distance < 1)
        if done: 
            reward = 100

        obs = self._get_obs()
        info = {'distance': distance}
        return obs, reward, done, info

    def _reset(self):
        """
        Returns:
            The first observation (dependent on obs_type)
        """        
        super(OneStaticCircleTarget, self)._reset()
        return self._get_obs()

    def _get_obs(self):
        if self._obs_type == 'OneDimCoord':
            return TwoToOneDim(self.target.image_coord, IMAGE_DIM)
        elif self._obs_type == 'image':
            img = self._get_image()
        return img

    def _get_image(self):
        return self.img

###############################################################################
#                                   Helpers                                   #
###############################################################################

class CircleTarget(object):
    def __init__(self, image_coord=(100,100), radius=20):
        self.image_coord = image_coord
        self.radius = radius

    def destroy(self):
        self.image_coord = None
        self.radius = None

    def rand_pos(self):
        self.image_coord = (np.random.randint(VIDEO_W), np.random.randint(VIDEO_H)) # random target pos within bounds

    def rand_size(self):
        self.radius = np.random.randint(MAX_TARGET_RADIUS)

    def virtual_target_position(self, env):
        x = self.image_coord[0] * (env.virtual_width / VIDEO_W)
        y = self.image_coord[1] * (env.virtual_height / VIDEO_H)
        return x,y

def OneToTwoDim(i, dimensions):
    """ 1D --> 2D
    Alt name could be R1toR2(R1point, R2dimensions)
    """
    return i % dimensions[0], i / dimensions[0] # (x,y)

def TwoToOneDim(xycoord, dimensions):
    """ 2D --> 1D
    Args:    xycoord     : tuple
             dimensions  : tuple with image width and height
    """
    return xycoord[0] + (dimensions[0] * xycoord[1])
    
###############################################################################
#                                     Main                                    #
###############################################################################

if __name__ == '__main__':
    env = OneStaticCircleTarget()
    # env._reset()

    try:
        while True:
            env._render()
    except KeyboardInterrupt:
        print 'interrupted!'
