#!/usr/bin/env python

# nerfbot_env.py
# Gerik Zatorski

import gym
from gym import spaces
from gym.utils import seeding
import gym_nerfbot

import numpy as np
import math

# for rendering
from gym.envs.classic_control import rendering
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)
import pyglet
from pyglet import gl

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
        # todo: do I need to seed here ...
        # self._seed()

        self.viewer = None
        self._obs_type = obs_type
        self.distance = distance                          # characterize distance from Nerfbot to the simulated wall
        self.motor_limits = (28, 28)                      # limit stepper motor ranges for safety
        self.home = (self.motor_limits[0]/2 ,             # calibrate for center of observation space
                     self.motor_limits[1]/2)
        self.state = self.home                            # set state to calibrated position

        # assumption: camera border is equivalent to motor aim at limits
        self.step_angle = math.radians(1.8) # 1.8 deg per step
        self.virtual_width = self.distance * math.tan(self.step_angle * self.motor_limits[0])
        self.virtual_height = self.distance * math.tan(self.step_angle * self.motor_limits[0])

    def _virtual_motor_pos(self):
        """ Simlate a shot's position based on Nerfbot's pan and tilt steps to a position on a virtual wall (meters?)
        TODO: should I check that it is within the limits?
        """
        horiz = self.virtual_width/2 + (self.distance * math.tan(self.step_angle * (self.state[0] - self.home[0])))
        vert = self.virtual_height/2 + (self.distance * math.tan(self.step_angle * (self.state[1] - self.home[1])))
        return horiz, vert

    def _reset(self):
        """ OpenAI gym API: reset
        Returns: initial observation
        """
        self.human_render = False
        # subclasses determine returned observation

    def _step(self, action):
        """ OpenAI gym API: step
        Override in ALL subclasses
        """
        raise NotImplementedError

    def _render(self, mode='human', close=False):
        """ OpenAI gym API: reset
        Based off OpenAI CarRacing-v0
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
                return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.transform = rendering.Transform()

        # draw subclass information (ex. the circle target)
        self.draw_image()

        arr = None
        win = self.viewer.window
        if mode != 'observation_pixels':
            self.draw_aim() # todo: move this to a better spot
            win.switch_to()
            win.dispatch_events()
        if mode=="rgb_array" or mode=="observation_pixels":
            win.clear()
            t = self.transform
            if mode=='rgb_array':
                VP_W = VIDEO_W
                VP_H = VIDEO_H
            else:
                VP_W = VIDEO_W
                VP_H = VIDEO_H
            gl.glViewport(0, 0, VP_W, VP_H)
            t.enable()
            # todo: self._render_noise()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]

        if mode=="rgb_array" and not self.human_render: # agent can call or not call env.render() itself when recording video.
            win.flip()

        if mode=='human':
            self.draw_aim() # todo: move this to a better spot
            self.human_render = True
            win.clear()
            t = self.transform
            gl.glViewport(0, 0, VIDEO_W, VIDEO_H)
            t.enable()
            # todo: self._render_noise()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            win.flip()

        return arr

    def _render_noise(self):
        """ TODO """
        raise NotImplementedError

    def draw_aim(self):
        t = rendering.Transform(translation=(self.state[0]*VIDEO_W/self.motor_limits[0], self.state[1]*VIDEO_H/self.motor_limits[1]))
        self.viewer.draw_circle(radius=8, res=30, filled=True, color=(0.0,0.8,0.0)).add_attr(t)

    # def _render(self, mode='human', close=False):
    #     """ Viewer only supports human mode """
    #     if close:
    #         if self.viewer is not None:
    #             os.kill(self.viewer.pid, signal.SIGKILL)
    #     else:
    #         if self.viewer is None:
    #             self._start_viewer()

    # def _destroy(self):
    #     if self.target:
    #         self.target.destroy()

###############################################################################
#                             Nerfbot Environments                            #
###############################################################################

class OneStaticCircleTarget(NerfbotEnv):
    """ OpenAI environment with one target
    Because the target is static, no update is need through observation

    TODO:
        * implement more shapes
    """
    def __init__(self, random=True, shape='Circle'):
        super(OneStaticCircleTarget, self).__init__()

        # spaces are a 1 dimensions representation of pan*tilt and image width*height
        self.action_space = spaces.Discrete(self.motor_limits[0] * self.motor_limits[1])
        if self._obs_type =='OneDimCoord':
            self.observation_space = spaces.Discrete(OBSERVATION_W * OBSERVATION_H)
        
        self.target = CircleTarget()

    def _step(self, action):

        assert self.action_space.contains(action)
        done = False

        self.state = OneToTwoDim(action, self.motor_limits)

        reward = 0
        aim_horiz, aim_vert = self._virtual_motor_pos()

        target_horiz, target_vert = self.target.virtual_target_position(self)

        dx = target_horiz - aim_horiz
        dy = target_vert - aim_vert
        distance = math.sqrt(dx**2 + dy**2)
        reward = -distance

        done = bool(distance < 2)
        if done: 
            reward = 100

        return TwoToOneDim(self.target.image_coord, IMAGE_DIM), reward, done, {}

    def _reset(self):
        """ 
        Depending on _obs_type, returns initial observation:
            ('OneDimCoord') - a 1D representation of the target center
        """
        super(OneStaticCircleTarget, self)._reset()
        if self._obs_type == 'OneDimCoord':
            return TwoToOneDim(self.target.image_coord, IMAGE_DIM)

    def draw_image(self):
        self.target.draw(self.viewer)

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

    # def rand_pos(self):
    #     self.image_coord = (np.random.randint(VIDEO_W), np.random.randint(VIDEO_H)) # random target pos within bounds

    # def rand_size(self):
    #     self.radius = np.random.randint(MAX_TARGET_RADIUS)

    def draw(self, viewer):
        t = rendering.Transform(translation=self.image_coord)
        viewer.draw_circle(radius=self.radius, res=30, filled=True, color=(0.8,0.0,0.0)).add_attr(t)

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
