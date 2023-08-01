import argparse

from ToSim import SimModel
from Controller import MouseController
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import deque
import scipy.io as scio
from Tools.PlotTool import plot_simple


# SCENENAME = "../models/dynamic_4l_t3.xml"
SCENENAME = "../models/scene_test1.xml"

# RUN_STEPS = 10000
RUN_STEPS = 4000


class DATA_Recorder():
    def __init__(self):
        self.imu_pos = deque([])
        self.imu_quat = deque([])
        self.imu_vel = deque([])
        self.imu_acc = deque([])
        self.imu_gyro = deque([])
        self.curtep = deque([])

    def update(self, theMouse):
        self.imu_pos.append(theMouse.pos)
        self.imu_quat.append(theMouse.quat)
        self.imu_vel.append(theMouse.vel)
        self.imu_acc.append(theMouse.acc)
        self.imu_gyro.append(theMouse.gyro)

    def savePath2(self, FileName):
        # scio.savemat('mvpath.mat', {'H_range': self.movePath})  # 写入mat文件
        imu_pos = np.array(self.imu_pos)
        imu_quat = np.array(self.imu_quat)
        imu_vel = np.array(self.imu_vel)
        imu_acc = np.array(self.imu_acc)
        imu_gyro = np.array(self.imu_gyro)
        scio.savemat(FileName + '.mat', {'pos': imu_pos, 'quat': imu_quat, 'vel': imu_vel,
                                         'acc': imu_acc, 'gyro': imu_gyro})  # 写入mat文件


def ProcessYH(RatEnv):
    # '''
    fig, axs = plt.subplots(2, 2)
    subTitle = ["Fore Left Leg", "Fore Right Leg",
                "Hind Left Leg", "Hind Right Leg"]
    for i in range(4):
        pos_1 = int(i / 2)
        pos_2 = int(i % 2)
        axs[pos_1, pos_2].set_title(subTitle[i])
        axs[pos_1, pos_2].plot(RatEnv.theController.trgXList[i], RatEnv.theController.trgYList[i])
        axs[pos_1, pos_2].plot(RatEnv.theMouse.legRealPoint_x[i], RatEnv.theMouse.legRealPoint_y[i])

    plt.show()

    # '''
    plt.plot(RatEnv.theController.trgXList[0], RatEnv.theController.trgYList[0], label='Target trajectory')
    plt.plot(RatEnv.theMouse.legRealPoint_x[0], RatEnv.theMouse.legRealPoint_y[0], label='Real trajectory ')
    plt.legend()
    plt.xlabel('y-coordinate (m)')
    plt.ylabel('z-coordinate (m)')
    plt.grid()
    plt.show()


class Simple_Wrapper():
    def __init__(self):
        super(Simple_Wrapper, self).__init__()
        parser = argparse.ArgumentParser("Description.")
        parser.add_argument('--fre', default=0.67,
                            type=float, help="Gait stride")
        args = parser.parse_args()

        self.theMouse = SimModel(SCENENAME, Render=True)
        self.theController = MouseController(args.fre, )
        for i in range(500):
            ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0, -1.2, 0, 0, 0, 0]
            self.theMouse.runStep(ctrlData)
        self.theMouse.initializing()
        # while True:
        #     self.theMouse.viewer.render()
        self.States_Init()

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
        if delta_vel<-0.2:
            self.Rewards_Attach = deque(-0.6 * np.ones(self.Window_Cover), maxlen=self.Window_Cover)
            # 如果State里面包含reward attach，那么需要对S[0]做更新


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
        done = False  # 超过了一定步数就重置一下环境
        info = None
        return s, r, done, info


if __name__ == '__main__':
    RatEnv = Simple_Wrapper()
    Record = DATA_Recorder()
    R = []
    V_vels = []
    Delta_vels = []

    start = time.time()
    for i in range(RUN_STEPS):
        # First get Theta
        # a  <- get a

        action = [1., 1., 1., 1.]
        tCtrlData = RatEnv.theController.runStep()  # No Spine
        # tCtrlData = theController.runStep_spine()		# With Spine
        ctrlData = tCtrlData

        RatEnv.theMouse.runStep(ctrlData, render=True)
        RatEnv.Action_Pre = action

        RatEnv.OneStepProcess()
        s, r, done, info = RatEnv.GetMarkovNode()
        print(s)

        Record.update(RatEnv.theMouse)
        R.append(r)
        V_vels.append(RatEnv.V_vel)
        Delta_vels.append(RatEnv.Delta_vel)

    end = time.time()
    timeCost = end - start
    print("Time -> ", timeCost)
    dis = RatEnv.theMouse.drawPath()
    print("py_v --> ", dis / timeCost)
    print("sim_v --> ", dis / (RUN_STEPS * 0.002))
    # theMouse.savePath("own_125")
    # Record.savePath2("imu_record_Scene1_1223")
    # ProcessYH(RatEnv)

    vel = np.array(Record.imu_vel)
    vel = np.transpose(vel)
    vely = -vel[1]

    N_StepPerT = 373
    Window_V = int(N_StepPerT / 2)
    plot_simple([vely[0:3000]*Window_V])

    plot_simple([V_vels[100:3000], Delta_vels[100:3000],
                 R[100+RatEnv.Window_Cover:3000++RatEnv.Window_Cover]])