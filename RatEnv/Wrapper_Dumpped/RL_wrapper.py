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
    def __init__(self, SceneName, Render=False):
        super(RatRL, self).__init__()
        # self.action_space = spaces.
        # self.observation_space=
        self.SceneName = SceneName
        # self.reset(Render=Render) should reset yourself

        # Wrapper
        high = np.array([np.inf] * 17).astype(np.float32)
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1]).astype(np.float32),
        )
        self.observation_space = spaces.Box(-high, high)
        self._max_episode_steps = 10000

        parser = argparse.ArgumentParser("Description.")
        parser.add_argument('--fre', default=0.67,
                            type=float, help="Gait stride")
        args = parser.parse_args()
        self.theMouse = SimModel(self.SceneName, Render=Render)
        self.theController = MouseController(args.fre)

    def reset(self):
        """将环境重置为初始状态，并返回一个初始状态；在环境中有随机性的时候，需要注意每次重置后要保证与之前的环境相互独立
        """
        self.theMouse.sim.set_state(self.theMouse.sim_state)

        self._step = 0
        for i in range(500):
            ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0, -1.2, 0, 0, 0, 0]
            self.theMouse.runStep(ctrlData)  # 此处有个内置的render
        self.theMouse.initializing()
        self.States_Init()
        print("Reset")

        # 是否需要hot up？ TODO

        for i in range(self.Window_Cover):
            s, _, _, _ = self.step([-1., -1., -1., -1.])
        return s

    def States_Init(self):
        self.N_StepPerT = 373
        self.Window_V = int(self.N_StepPerT / 2)
        self.Window_Delta = int(self.N_StepPerT / 8)
        self.Window_Cover = int(self.N_StepPerT / 4)

        self.V_vel_deque = deque(np.zeros(self.Window_Delta), maxlen=self.Window_Delta)
        # self.Delta_vels = []

        self.vel_deque = deque(np.zeros(self.Window_V), maxlen=self.Window_V)

        self.Rewards_Base = deque(np.zeros(self.Window_Cover), maxlen=self.Window_Cover)
        self.Rewards_Attach = deque(np.zeros(self.Window_Cover), maxlen=self.Window_Cover)

        # For States
        self.action = [0, 0, 0, 0]
        self.State_deque = deque([], maxlen=self.Window_Cover)
        self.V_vel = 0.
        self.Delta_vel = 0.
        self.Action_Pre = [1., 1., 1., 1.]

    def render(self, mode='human'):
        self.theMouse.viewer.render()

    def close(self):
        """一些环境数据的释放可以在该函数中实现
        """
        pass

    def seed(self, seed=None):
        """设置环境的随机生成器
        """
        return

    def OneStepProcess(self):
        # Cut
        self.vel_deque.append(-list(self.theMouse.vel)[1])  # Update v
        V_vel = sum(self.vel_deque)

        k_ins = 1 / 50
        k_bias = -0.05
        V_vel = V_vel * k_ins + k_bias
        self.V_vel = V_vel
        V_vel_pre =self.V_vel_deque[0]
        self.V_vel_deque.append(V_vel)

        # Cal Delta Func of Vel
        delta_vel = (V_vel - V_vel_pre) * 100 / self.Window_Delta
        # self.Delta_vels.append(delta_vel)
        self.Delta_vel = delta_vel

        # Reward Append
        self.Rewards_Base.append(V_vel)
        # State FIFO
        self.StateFIFO()

        # Trigger of Reward Attach -- punishment
        self.Rewards_Attach.append(0.0)
        if delta_vel < -0.2:
            self.Rewards_Attach = deque(-0.6 * np.ones(self.Window_Cover), maxlen=self.Window_Cover)
            # 如果State里面包含reward attach，那么需要对S[0]做更新 TODO

    def StateFIFO(self):
        vel = self.theMouse.vel
        acc = self.theMouse.acc
        gyro = self.theMouse.gyro
        r_base = self.Rewards_Base[self.Rewards_Base.maxlen-1]
        # r_attach = self.Rewards_Attach[self.Rewards_Attach.maxlen-1]

        # SelfInfo of theta
        theta = 2 * np.pi * (self.theController.curStep) / self.theController.SteNum
        # SelfInfo of t action TODO

        S = [[theta], self.Action_Pre,
             [r_base, self.V_vel, self.Delta_vel],
             vel, acc, gyro]
        S = [y for x in S for y in x]
        S = np.array(S)
        self.State_deque.append(S)
        # S_Out = self.State_deque[0]

    def GetMarkovNode(self):
        # 获得的是 Window_Cover 时刻之前的
        r = self.Rewards_Base[0] + self.Rewards_Attach[0]
        s = self.State_deque[0]

        if self._step < self._max_episode_steps:
            done = False # 超过了一定步数就重置一下环境
        else:
            done = True
        info = None
        return s, r, done, info

    def step(self, action):
        """环境的主要驱动函数，主逻辑将在该函数中实现。该函数可以按照时间轴，固定时间间隔调用

        参数:
            action (object): an action provided by the agent

        返回值:
            observation (object): agent对环境的观察，在本例中，直接返回环境的所有状态数据
            reward (float) : 奖励值，agent执行行为后环境反馈
            done (bool): 该局游戏时候结束，在本例中，只要自己被吃，该局结束
            info (dict): 函数返回的一些额外信息，可用于调试等
        """
        # ac = [] # rho, theta
        ActionSignal = (action + 1) / 2
        tCtrlData = self.theController.runStep(ActionSignal)  # No Spine
        # tCtrlData = theController.runStep_spine()		# With Spine
        ctrlData = tCtrlData
        self.theMouse.runStep(ctrlData)
        self._step = self._step + 1

        self.Action_Pre = action
        self.OneStepProcess()
        s, r, done, info = self.GetMarkovNode()

        return s, r, done, info