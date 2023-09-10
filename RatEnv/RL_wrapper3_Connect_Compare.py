import gym
from gym import spaces
import argparse

from mujoco_py import load_model_from_path, MjSim, MjViewer
# from RatEnv.ToSim import SimModel
from RatEnv.RL_Controller import MouseController
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import deque


class RatRL(gym.Env):
    def __init__(self, xml_file, Render=False):
        super(RatRL, self).__init__()
        # Wrapper
        high = np.array([np.inf] * 19).astype(np.float32)
        self.action_space = spaces.Box(
            np.array([-1., -1., -1., -1., -1., -1., -1., -1.]).astype(np.float32),
            np.array([+1., +1., +1., +1., +1., +1., +1., +1.]).astype(np.float32),
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

        self.sim_state = self.sim.get_state()
        self.sim.set_state(self.sim_state)
        self.imu_pos = deque([])
        self.imu_quat = deque([])
        self.imu_vel = deque([])
        self.imu_acc = deque([])
        self.imu_gyro = deque([])
        # self.theMouse = SimModel(xml_file, Render=Render)

        # Controller
        self.frame_skip = 1 # 5
        self._timestep = 0.01  # Default
        self.model.opt.timestep = self._timestep  # Default = 0.002s per timestep
        self.dt = self._timestep * self.frame_skip  # dt = 0.01s
        self._max_episode_steps = int(10000 * 0.002 / self.dt)  # 10000 for 0.002

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            # mujoco.mj_step(self.model, self.data)
            self.sim.step()
        if self.Render:
            # self.viewer.sync()
            self.viewer.render()

    def reset(self):
        """将环境重置为初始状态，并返回一个初始状态；在环境中有随机性的时候，需要注意每次重置后要保证与之前的环境相互独立
        """
        # self.theMouse.sim.set_state(self.theMouse.sim_state)  # Go to initializing
        self.ActionIndex = 0
        self.sim.set_state(self.sim_state)
        self.imu_pos = deque([])
        self.imu_quat = deque([])
        self.imu_vel = deque([])
        self.imu_acc = deque([])
        self.imu_gyro = deque([])

        self._step = 0
        for i in range(500):
            ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0, -1.2, 0, 0, 0, 0]
            self.do_simulation(ctrlData, n_frames=1)  # 此处有个内置的render
        self.pos = list(self.sim.data.sensordata[16:16 + 3])  # imu_pos
        self.quat = list(self.sim.data.sensordata[19:19 + 4])
        self.vel = list(self.sim.data.sensordata[23:23 + 3])
        self.acc = list(self.sim.data.sensordata[26:26 + 3])
        self.gyro = list(self.sim.data.sensordata[29:29 + 3])
        self.imu_pos.append(self.pos)

        # States Init
        # self.posY_Dir_pre = 0
        # Connect Version
        # self.vel_list = []
        # self.vels = deque([])
        # self.gyros = deque([])
        self.action = [0.]*8
        self.Action_Pre = [0.]*8
        self.Y_Pre = self.pos[1]

        # 是否需要hot up？ TODO
        action_hot = [0.] * 8
        self.done = False
        s, _, _, _ = self.step(action_hot)
        # print("Reset")
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
            action (object): an action provided by the agent

        Return:
            observation (object): agent对环境的观察，在本例中，直接返回环境的所有状态数据
            reward (float) : 奖励值，agent执行行为后环境反馈
            done (bool): 该局游戏时候结束，在本例中，只要自己被吃，该局结束
            info (dict): 函数返回的一些额外信息，可用于调试等
        """
        self.Action_Pre = action
        self.Y_Pre = self.pos[1]

        self.action = np.array(action)
        ActionSignal = np.array(action) * 1.57  # np.pi/2
        ctrlData = list(ActionSignal)
        ctrlData.extend([0] * 4)
        self.do_simulation(ctrlData, n_frames=self.frame_skip)
        # imudata
        self.pos = list(self.sim.data.sensordata[16:16 + 3])  # imu_pos
        self.quat = list(self.sim.data.sensordata[19:19 + 4])
        self.vel = list(self.sim.data.sensordata[23:23 + 3])
        self.acc = list(self.sim.data.sensordata[26:26 + 3])
        self.gyro = list(self.sim.data.sensordata[29:29 + 3])
        self.imu_pos.append(self.pos)

        # vel_dir = -list(self.vel)[1]
        # self.vel_list.append(vel_dir)
        # self.vels.append(vel)
        # self.gyros.append(gyro)

        # reward = -vel[1] * 4
        reward_forward = (self.pos[1] - self.Y_Pre) / self.dt * (-5) # 2~4

        # self.rat = self.FFTProcess(np.array(self.gyros).transpose()[1])
        # if self.rat < 0.6:
        #     reward = reward - 0.3
        reward_trapped = 0.0
        if self.pos[2] < 0.04:
            reward_trapped = -5.0  # 1.5?   15.0 Too Large For 51   5.0 For 52
            self.done = True

        self.Reward_Now = reward_forward + reward_trapped

        S = [self.action,
             [self.Reward_Now],
             self.vel, self.gyro, self.quat]
        # (8) + (1) + (3 + 3 + 4) = 19
        #      self.pos, self.vel, self.acc, self.gyro, self.quat]
        S = [y for x in S for y in x]
        S = np.array(S).astype(np.float32)
        # S = np.array(S)
        # S = self.sim.data.sensordata.copy().astype(np.float32)  # SensorDir
        self.State_Now = S

        if self._step > self._max_episode_steps:
            self.done = True  # 超过了一定步数就重置一下环境
        info = {
            "reward_forward": reward_forward,
            # "reward_bias": reward_bias,
            # "reward_holdon": reward_holdon,
            # "sum_delta_a": sum_delta_a,
            # "touch": contact_sensor
        }
        # self.vel_list = []
        # self.vels = deque([])
        # self.gyros = deque([])

        self._step = self._step + 1
        return self.State_Now, self.Reward_Now, self.done, info


if __name__ == '__main__':
    RENDER = True

    RUN_STEPS = 4000
    # RUN_STEPS = 50  # Half Per Action
    SceneName = "../models/dynamic_4l_t3.xml"
    # SceneName = "../models/dynamic_4l_t3_Change.xml"
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
        action = [0, 0, 0, 0, 0, 0, 0, 0]  # FL, FR, HL, HR  [-1, 1]-->[0,1]:  (a+1)/2
        observation, reward, done, info = env.step(action)
        # V_vels.append(-env.Vels_mean[1]*4)
        # V_x.append(env.Vels_mean[0])
        # V_z.append(env.Vels_mean[2])
        # Delta_vels.append(env.Delta_vel)
        # R.append(reward)
        # print(env.theController.curStep)
        # plot_simple([info])

    # plot_simple([V_vels[100:3000], Delta_vels[100:3000],
    #              R[100 + env.Window_Cover:3000 + +env.Window_Cover]])

    # plot_simple([V_vels[:], R[:], Rats],
    #             leg=['V_vels', 'R', 'rats'])

    # plot_simple([V_x, V_z],
    #             leg=['V_x', 'V_z'])


