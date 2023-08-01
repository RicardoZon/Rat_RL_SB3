import gym
from gym import spaces
import argparse

from RatEnv.ToSim import SimModel
from RatEnv.RL_Controller import MouseController
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import deque

class RatRL(gym.Env):
    """
    Sparse Version
    Action: angle of 8 motor
    """
    def __init__(self, SceneName, Render=False, timestep=0.002):
        super(RatRL, self).__init__()
        # self.action_space = spaces.
        # self.observation_space=
        self.SceneName = SceneName
        # self.reset(Render=Render) should reset yourself

        # Wrapper
        high = np.array([np.inf] * 16).astype(np.float32)
        self.action_space = spaces.Box(
            np.array([-1.57, -1.57, -1.57, -1.57, -1.57, -1.57, -1.57, -1.57]).astype(np.float32),
            np.array([+1.57, +1.57, +1.57, +1.57, +1.57, +1.57, +1.57, +1.57]).astype(np.float32),
        )
        # self.action_space = spaces.Box(
        #     np.array([-1., -1., -1., -1., -1., -1., -1., -1.]).astype(np.float32),
        #     np.array([+1., +1., +1., +1., +1., +1., +1., +1.]).astype(np.float32),
        # )
        self.observation_space = spaces.Box(-high, high)

        self.theMouse = SimModel(self.SceneName, Render=Render)
        self._timestep = self.theMouse.model.opt.timestep
        self.frame_skip = 5
        self.dt = self._timestep * self.frame_skip
        self._max_episode_steps = 10000*0.002/self.dt  # 10000 for 0.002

        parser = argparse.ArgumentParser("Description.")
        parser.add_argument('--fre', default=0.67,
                            type=float, help="Gait stride")
        args = parser.parse_args()
        self.theController = MouseController(args.fre, dt=self.dt)

    def reset(self):
        """将环境重置为初始状态，并返回一个初始状态；在环境中有随机性的时候，需要注意每次重置后要保证与之前的环境相互独立
        """
        self.theMouse.initializing()
        self._step = 0
        for i in range(500):
            ctrlData = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]
            self.theMouse.runStep(ctrlData)  # 此处有个内置的render
        self.theMouse.initializing()
        self.theController.reset()
        self.States_Init()

        # 是否需要hot up？ TODO
        action_hot = [0., 0., 0., 0., 0., 0., 0., 0.]
        self.done = False
        s, _, _, _ = self.step(action_hot)
        return s

    def States_Init(self):
        # self.N_StepPerT = 373
        # For States
        self.action = [0., 0., 0., 0., 0., 0., 0., 0.]
        self.Action_Pre = [0., 0., 0., 0., 0., 0., 0., 0.]

    def render(self, mode='human'):
        self.theMouse.viewer.render()

    def close(self):
        """
        """
        pass

    def seed(self, seed=None):
        """设置环境的随机生成器
        """
        return

    def OneStepProcess(self):
        # Cut
        # vel_dir = -list(self.theMouse.vel)[1]
        # pos = self.theMouse.pos
        # posY_Dir = -pos[1]
        vel = self.theMouse.vel
        gyro = self.theMouse.gyro

        self.Vels_mean = vel
        self.Gyros_mean = gyro
        reward_forward = vel[1] * (-4)  # Dir


        reward_trapped = 0.0
        if self.theMouse.pos[2] < 0.04:
            reward_trapped = -0.1
            self.done = True

        # 0.1 36   .05  37 0.5 38
        control_cost = 0.1 * np.sum(
            np.square(self.Action_Pre)
        )

        self.Reward_Now = float(reward_forward + reward_trapped - control_cost)  # FLOAT 32

        # SelfInfo of theta
        self.ActionIndex = (self.theController.curStep) / self.theController.SteNum

        S = [self.Action_Pre, [self.ActionIndex],
             [self.Reward_Now],
             vel, gyro]
        S = [y for x in S for y in x]
        S = np.array(S).astype(np.float32)
        self.State_Now = S
        # self.State_deque.append(S)

    def GetMarkovNode(self):
        r = self.Reward_Now
        s = self.State_Now
        if self._step > self._max_episode_steps:
            self.done = True # 超过了一定步数就重置一下环境
            # print("Out")
        info = {}
        return s, r, self.done, info

    def step(self, action, Render=True, LegCal=False):
        # action x8 [-1, +1]
        ActionSignal = np.array(action) * np.pi /2

        ctrlData = list(ActionSignal)
        for i in range(4):
            ctrlData.append(0)  # Append Spine, head, tails.

        for _ in range(self.frame_skip):
            self.theMouse.runStep(ctrlData, legposcal=LegCal)
            if Render:
                self.render()

        self._step = self._step + 1
        self.Action_Pre = action
        self.OneStepProcess()
        s, r, done, info = self.GetMarkovNode()

        return s, r, done, info