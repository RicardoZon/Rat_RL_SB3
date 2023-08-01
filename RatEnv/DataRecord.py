import argparse

from ToSim import SimModel
from Controller import MouseController
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import deque
import scipy.io as scio
from Tools.PlotTool import plot_simple
from RatEnv.RL_wrapper3_Connect import RatRL


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



if __name__ == '__main__':
    RatEnv = RatRL("SceneName", Render=False)
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