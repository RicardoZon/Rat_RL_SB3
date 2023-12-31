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

# For Validation 57, 60
class SimModel(object):
    def __init__(self, xml_file: str, Render=False):
        super(SimModel, self).__init__()
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

    def initializing(self):
        self.sim.set_state(self.sim_state)

        self.imu_pos = deque([])
        self.imu_quat = deque([])
        self.imu_vel = deque([])
        self.imu_acc = deque([])
        self.imu_gyro = deque([])

    def runStep(self, ctrlData, legposcal=False):
        self.sim.data.ctrl[:] = ctrlData
        self.sim.step()

        # imudata
        self.pos = list(self.sim.data.sensordata[16:16 + 3])  # imu_pos
        self.quat = list(self.sim.data.sensordata[19:19 + 4])
        self.vel = list(self.sim.data.sensordata[23:23 + 3])
        self.acc = list(self.sim.data.sensordata[26:26 + 3])
        self.gyro = list(self.sim.data.sensordata[29:29 + 3])

        self.imu_pos.append(self.pos)


class RatRL(gym.Env):
    def __init__(self, xml_file, Render=False, Recorder=None):
        super(RatRL, self).__init__()
        # Wrapper
        high = np.array([np.inf] * 12).astype(np.float32)
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1]).astype(np.float32),
        )
        self.observation_space = spaces.Box(-high, high)
        # self._max_episode_steps = 50
        # self._max_episode_steps = 50*2  # V3_1 T/4
        self._max_episode_steps = 50 * 4  # V3_2 T/8  100*
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
        self.frame_skip = 1
        self._timestep = self.model.opt.timestep  # Default = 0.002s per timestep
        self.dt = self._timestep * self.frame_skip  # dt = 0.01s
        fre = 0.67
        self.theController = MouseController(fre, dt=self.dt)
        self.ActionIndex = 0
        self.Action_Div = [47, 47, 47, 47, 47, 47, 47, 47]  # 93, 93, 93, 94
        self.MaxActIndex = len(self.Action_Div)

        # Recorder
        if Recorder:
            self.Recorder = Recorder

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

            self.imu_pos.append(self.pos)

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
        #  sth in initializing should be done TODO
        self.theController.curStep = 0  # Controller Reset
        self.theController.trgXList = [[], [], [], []]
        self.theController.trgYList = [[], [], [], []]

        # States Init
        self.posY_Dir_pre = 0
        # Connect Version
        self.vel_list = []
        self.vels = deque([])
        self.gyros = deque([])
        self.action = [1., 1., 1., 1.]
        self.Action_Pre = [1., 1., 1., 1.]

        # 是否需要hot up？ TODO
        action_hot = [1., 1., 1., 1.]
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
        # ac = [] # rho, theta

        # 一次执行一个half周期
        index = self.ActionIndex
        for _ in range(self.Action_Div[index]):
            ActionSignal = (np.array(action) + 1.0) / 2

            tCtrlData = self.theController.runStep(ActionSignal)  # No Spine

            vel_dir = -list(self.vel)[1]
            self.vel_list.append(vel_dir)
            vel = self.vel
            self.vels.append(vel)
            gyro = self.gyro
            self.gyros.append(gyro)

            self.do_simulation(tCtrlData, n_frames=self.frame_skip)

            if self.Recorder:
                # manually updata recorder
                # TODO: write it as a callback
                self.Recorder.update(self)
                self.Recorder.ctrldata.append(self.sim.data.ctrl.copy())

        # pos = self.theMouse.pos
        # posY_Dir = -pos[1]
        # reward = posY_Dir - self.posY_Dir_pre
        # self.posY_Dir_pre = posY_Dir
        #
        # self.Reward_Now = reward*10

        vels_mean = np.array(self.vels).mean(axis=0)
        gyros_mean = np.array(self.gyros).mean(axis=0)

        reward = -vels_mean[1] * 4

        # self.rat = self.FFTProcess(np.array(self.gyros).transpose()[1])
        # if self.rat < 0.6:
        #     reward = reward - 0.3

        self.Reward_Now = reward

        S = [self.Action_Pre, [self.ActionIndex],
             [reward],
             vels_mean, gyros_mean]
        # (4+1) + (1) + (3 +3) = 12
        S = [y for x in S for y in x]
        # S = np.array(S).astype(np.float32)
        S = np.array(S)
        self.State_Now = S

        # Optional for test
        self.Vels_mean = vels_mean
        self.Gyros_mean = gyros_mean

        self.ActionIndex = (self.ActionIndex + 1) % self.MaxActIndex  # Go to next Action Piece

        self._step = self._step + 1
        self.Action_Pre = action

        # r = self.Rewards_Base[0] + self.Rewards_Attach[0]
        # s = self.State_deque[0]
        if self._step > self._max_episode_steps:
            self.done = True  # 超过了一定步数就重置一下环境
            # print("Out")
        # info = None
        info = {
            "ActionIndex": self.ActionIndex,
            # "reward_bias": reward_bias,
            # "reward_holdon": reward_holdon,
            # "sum_delta_a": sum_delta_a,
            # "touch": contact_sensor
        }
        # info = self.vel_list
        # print(np.mean(info))

        self.vel_list = []
        self.vels = deque([])
        self.gyros = deque([])

        return self.State_Now, self.Reward_Now, self.done, info

    def FFTProcess(self, data, Div=16, show=False, leg=None):
        T = 0.02
        Fs = 1 / T  # 采样频率
        L = len(data)
        n = L
        # if Div is None:
        #     Div = 25
        ncut = int(n / Div)  # 50/25 = 2 Hz
        f = np.linspace(0, Fs, n)

        out = np.fft.fft(data)
        power = abs(out) ** 2
        fcut = f[0:ncut]
        power = power[0:ncut]

        rat = power[0] / sum(power)
        return rat


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

    Actions = [[-3, -3, 1, 1],
               [-3, -3, 1, 1]
    ]
    for _ in range(RUN_STEPS):
        # theta
        action = [1, 1, 1, 1]  # FL, FR, HL, HR  [-1, 1]-->[0,1]:  (a+1)/2
        observation, reward, done, info = env.step(action)
        V_vels.append(-env.Vels_mean[1]*4)
        V_x.append(env.Vels_mean[0])
        V_z.append(env.Vels_mean[2])
        # Delta_vels.append(env.Delta_vel)
        R.append(reward)
        print(env.theController.curStep)
        # plot_simple([info])

    # plot_simple([V_vels[100:3000], Delta_vels[100:3000],
    #              R[100 + env.Window_Cover:3000 + +env.Window_Cover]])

    # plot_simple([V_vels[:], R[:], Rats],
    #             leg=['V_vels', 'R', 'rats'])

    # plot_simple([V_x, V_z],
    #             leg=['V_x', 'V_z'])


