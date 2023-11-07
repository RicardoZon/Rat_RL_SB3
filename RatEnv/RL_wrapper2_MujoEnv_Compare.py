# 073  add contact cost  W_c = 0.1   V
# 074  add contact cost  W_c = 0.05
# 075  add contact cost  W_c = 0.01  X useless
import gym
from gym import spaces
import argparse
from mujoco_py import load_model_from_path, MjSim, MjViewer
from RatEnv.ToSim import SimModel
# from RatEnv.RL_Controller import MouseController
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import deque
from gym.envs.mujoco import mujoco_env  # REF


class RatRL(gym.Env):
    """
    Sparse Version
    Action: angle of 8 motor
    Simplified from Wrapper V2
    """

    def __init__(self, xml_file, Render=False, timestep=0.002):
        super(RatRL, self).__init__()
        # Wrapper
        high = np.array([np.inf] * 23).astype(np.float32)
        self.action_space = spaces.Box(
            np.array([-1., -1., -1., -1., -1., -1., -1., -1.]).astype(np.float32),
            np.array([+1., +1., +1., +1., +1., +1., +1., +1.]).astype(np.float32),
        )
        self.observation_space = spaces.Box(-high, high)
        self.model = load_model_from_path(xml_file)
        self.sim = MjSim(self.model)
        if Render:
            # render must be called mannually
            self.viewer = MjViewer(self.sim)
            self.viewer.cam.azimuth = 0
            self.viewer.cam.lookat[0] += 0.25
            self.viewer.cam.lookat[1] += -0.5
            self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.Render = Render

        self.init_state = self.sim.get_state()
        self.sim.set_state(self.init_state)

        self.frame_skip = 1
        self._timestep = 0.01  # Default 0.002
        self.model.opt.timestep = self._timestep  # Default = 0.002s per timestep
        self.dt = self._timestep * self.frame_skip
        self._max_episode_steps = 10000 * 0.002 / self.dt  # 10000 for 0.002

        self.pos = None
        self.quat = None
        self.vel = None
        self.acc = None
        self.gyro = None
        self._step = None
        self.nair = 0

    def do_simulation(self, ctrl, n_frames, Render=False):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()
            if self.Render:
                self.viewer.render()

    def reset(self):
        """将环境重置为初始状态，并返回一个初始状态；在环境中有随机性的时候，需要注意每次重置后要保证与之前的环境相互独立
        """
        self.sim.set_state(self.init_state)
        self._step = 0

        ctrlData = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]
        self.do_simulation(ctrlData, n_frames=5)  # 此处有个内置的render
        self.pos = list(self.sim.data.sensordata[16:16 + 3])  # com_pos from imu  # imu_pos
        self.action = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        self.done = False
        s, _, _, _ = self.step(self.action)
        return s

    def render(self, mode='human'):
        self.viewer.render()

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
        vel = self.vel
        gyro = self.gyro
        contact_sensor = [self.sim.data.get_sensor("fl_t1"), self.sim.data.get_sensor("fr_t1"),
                          self.sim.data.get_sensor("rl_t1"), self.sim.data.get_sensor("rr_t1")]
        contact_sensor = (np.array(contact_sensor) != 0.0).astype(int)  # 1 for contact, 0 for air
        sum_contact = sum(contact_sensor)

        # Rewards
        reward_forward = (self.pos[1] - self.Y_Pre) / self.dt * (-5)  # 2~4

        reward_trapped = 0.0
        if reward_forward > 3.5:
            reward_trapped = -5.0  # 71 72

        if sum_contact == 0:
            self.nair += 1
            # reward_trapped = -1.0  # weaken air time of front paws?
        else:
            self.nair = 0
        if self.nair == 10:  # 10 For 65 20 For 66
            # Trapped
            reward_trapped = -5.0  # 1.5?   15.0 Too Large For 51   5.0 For 52
            self.done = True
        # if self.pos[2] < 0.03:
        #     reward_trapped = -5.0  # 1.5?   15.0 Too Large For 51   5.0 For 52
        #     self.done = True
        reward_bias = 0.  # -(self.pos[0] * 2) **2  # For 47  *4 For 46
        # # reward_holdon = 0. # 1. * self._step / self._max_episode_steps  #  X
        # reward_height = 0. # 25 * (#self.pos[2]-0.065)  # make jump and trap

        sum_delta_a = sum(abs(self.action - self.action_pre))
        control_cost = 0.05 * sum_delta_a

        # qpos
        # qposes = [
        #     self.sim.data.get_joint_qpos("knee1_fl"),
        #     self.sim.data.get_joint_qpos("ankle_fl"),4
        #     self.sim.data.get_joint_qpos("knee1_fr"),
        #     self.sim.data.get_joint_qpos("ankle_fr"),
        #     self.sim.data.get_joint_qpos("knee1_rl"),
        #     self.sim.data.get_joint_qpos("ankle_rl"),
        #     self.sim.data.get_joint_qpos("knee1_rr"),
        #     self.sim.data.get_joint_qpos("ankle_rr"),
        # ]

        self.Reward_Now = float(reward_forward + reward_trapped + reward_bias - control_cost)  # FLOAT 32

        # SelfInfo of theta
        # self.ActionIndex = 0.
        S = [self.action,
             [reward_forward],
             vel, gyro, self.quat, contact_sensor]
        # (8) + (1) + (3+3+4+4) = 23
        # S = [self.action,
        #      [reward_forward],
        #      vel, gyro, self.quat, contact_sensor]
        S = [y for x in S for y in x]
        S = np.array(S).astype(np.float32)
        self.State_Now = S
        # print(S)

        # get markov
        r = self.Reward_Now
        s = self.State_Now
        if self._step > self._max_episode_steps:
            self.done = True  # 超过了一定步数就重置一下环境
            # print("Out")
        if self.pos[1] < -2.0:
            self.done = True  # For validation
        info = {
            "reward_forward": reward_forward,
            "reward_bias": reward_bias,
            # "reward_holdon": reward_holdon,
            "sum_delta_a": sum_delta_a,
            "touch": contact_sensor
        }
        # if not np.any(contact_sensor):
        # print("!!!!!!!!!! Zero!!!!!!!!!! Zero!!!!!!!!!! Zero!!!!!!!!!! Zero!!!!!!!!!! Zero!!!!!!!!!! Zero")

        return s, r, self.done, info

    def step(self, action, Render=False, LegCal=False):
        self.Y_Pre = self.pos[1]
        self.action_pre = self.action

        # action x8 [-1, +1]
        self.action = np.array(action)
        ActionSignal = np.array(action) * 1.57  # np.pi/2
        ctrlData = list(ActionSignal)
        for i in range(4):
            ctrlData.append(0)  # Append Spine, head, tails.

        self.do_simulation(ctrlData, n_frames=self.frame_skip, Render=Render)

        self.pos = list(self.sim.data.sensordata[16:16+3])  # com_pos from imu  # imu_pos  list()  or addr dependence
        self.quat = list(self.sim.data.sensordata[19:19 + 4])
        self.vel = list(self.sim.data.sensordata[23:23 + 3])
        self.acc = list(self.sim.data.sensordata[26:26 + 3])
        self.gyro = list(self.sim.data.sensordata[29:29 + 3])

        self._step = self._step + 1
        s, r, done, info = self.OneStepProcess()
        return s, r, done, info
