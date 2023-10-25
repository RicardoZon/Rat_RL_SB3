import gym
from gym import spaces
import argparse

from mujoco_py import load_model_from_path, MjSim, MjViewer
# from RatEnv.ToSim import SimModel
from RatEnv.Controller import MouseController
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import deque


class RatRL(gym.Env):
    def __init__(self, xml_file, fre_cyc=0.67, Render=False):
        super(RatRL, self).__init__()
        # Wrapper
        high = np.array([np.inf] * 24).astype(np.float32)
        self.action_space = spaces.Box(
            np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1.]).astype(np.float32),
            np.array([+1., +1., +1., +1., +1., +1., +1., +1., +1.]).astype(np.float32),
        )
        self.observation_space = spaces.Box(-high, high)
        self.Render = Render
        self.model = load_model_from_path(xml_file)
        self.sim = MjSim(self.model)
        if Render:
            # render must be called mannually
            self.viewer = MjViewer(self.sim)
            self.viewer.cam.azimuth = 0
            self.viewer.cam.lookat[0] += 0.25
            self.viewer.cam.lookat[1] += -0.5
            self.viewer.cam.distance = self.model.stat.extent * 0.5

        self.init_state = self.sim.get_state()
        self.sim.set_state(self.init_state)

        self.model.opt.timestep = 0.002
        self._timestep = self.model.opt.timestep
        self.frame_skip = 5  # 001 5
        self.dt = self._timestep * self.frame_skip
        self._max_episode_steps = 10000 * 0.002 / self.dt  # 10000 for 0.002

        # Controller
        self.fre_cyc = fre_cyc # 1.25  # 0.80?
        self.SteNum = int(1 / (self.dt * self.fre_cyc) / 2)  # /1.25)
        print("SteNum --> {}".format(self.SteNum))
        self.theController = MouseController(SteNum=self.SteNum)

        self.pos = None
        self.quat = None
        self.vel = None
        self.acc = None
        self.gyro = None
        self._step = None
        self.nair = 0

        self.imu_pos = []

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            # mujoco.mj_step(self.model, self.data)
            self.sim.step()
            # imudata
            self.pos = list(self.sim.data.sensordata[16:16 + 3])  # imu_pos
            self.quat = list(self.sim.data.sensordata[19:19 + 4])
            self.vel = list(self.sim.data.sensordata[23:23 + 3])
            self.acc = list(self.sim.data.sensordata[26:26 + 3])
            self.gyro = list(self.sim.data.sensordata[29:29 + 3])

        if self.Render:
            # self.viewer.sync()
            self.viewer.render()

    def reset(self):
        """将环境重置为初始状态，并返回一个初始状态；在环境中有随机性的时候，需要注意每次重置后要保证与之前的环境相互独立
        """
        self.sim.set_state(self.init_state)
        self._step = 0
        ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0, -1.2, 0, 0, 0, 0]
        self.do_simulation(ctrlData, n_frames=100)  # 此处有个内置的render

        self.theController.curStep = 0  # Controller Reset
        self.theController.trgXList = [[], [], [], []]
        self.theController.trgYList = [[], [], [], []]

        self.action = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.done = False
        s, _, _, _ = self.step(self.action)
        return s

    def render(self, mode='human'):
        # self.theMouse.viewer.render()
        pass

    def close(self):
        """一些环境数据的释放可以在该函数中实现
        """
        pass

    def seed(self, seed=None):
        """设置环境的随机生成器
        """
        return

    def step(self, action, LegCal=False):
        """
        Para:
            action (object): delta of angle

        Return:
            observation (object): agent对环境的观察，在本例中，直接返回环境的所有状态数据
            reward (float) : 奖励值，agent执行行为后环境反馈
            done (bool): 该局游戏时候结束，在本例中，只要自己被吃，该局结束
            info (dict): 函数返回的一些额外信息，可用于调试等
        """
        self.Y_Pre = self.pos[1]
        self.action_pre = self.action

        ranges = [1.57] * 8 + [0.8]
        delta = np.array(action) * ranges
        delta = np.insert(delta, 8, [0,0,0])
        ctrlData = self.theController.runStep()
        ctrlData = ctrlData + delta
        ctrlData[ctrlData > 1.57] = 1.57
        ctrlData[ctrlData < -1.57] = -1.57
        self.do_simulation(ctrlData, n_frames=self.frame_skip)

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
            self.done = True  # For validation Success
        info = {
            "reward_forward": reward_forward,
            "reward_bias": reward_bias,
            # "reward_holdon": reward_holdon,
            "sum_delta_a": sum_delta_a,
            "touch": contact_sensor
        }
        # if not np.any(contact_sensor):
        # print("!!!!!!!!!! Zero!!!!!!!!!! Zero!!!!!!!!!! Zero!!!!!!!!!! Zero!!!!!!!!!! Zero!!!!!!!!!! Zero")

        self._step = self._step + 1
        return self.State_Now, self.Reward_Now, self.done, info


if __name__ == '__main__':
    RENDER = True

    RUN_STEPS = 4000
    # RUN_STEPS = 50  # Half Per Action
    SceneName = "../models/dynamic_4l_t3.xml"
    # SceneName = "../models/Scenario1_Planks.xml"
    # SceneName = "../models/Scenario3_Logs.xml"
    # SceneName = "../models/Scenario4_Stairs.xml"

    env = RatRL(SceneName, Render=RENDER)
    R = []
    V_vels = []
    V_x = []
    V_z = []
    Rats = []
    s = env.reset()

    for _ in range(RUN_STEPS):
        # theta
        action = [0.0] * 8 + [0.0]
        # action = [1, 1, 1, 1, 1, 1, 1, 1, 0.9]  # FL, FR, HL, HR  [-1, 1]-->[0,1]:  (a+1)/2
        observation, reward, done, info = env.step(action)
        print(info)

        R.append(reward)
        # print(env.theController.curStep)
        # plot_simple([info])

    # plot_simple([V_vels[100:3000], Delta_vels[100:3000],
    #              R[100 + env.Window_Cover:3000 + +env.Window_Cover]])

    # plot_simple([V_vels[:], R[:], Rats],
    #             leg=['V_vels', 'R', 'rats'])

    # plot_simple([V_x, V_z],
    #             leg=['V_x', 'V_z'])


