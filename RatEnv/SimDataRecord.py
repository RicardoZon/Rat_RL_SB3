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
from RatEnv.ToSim import SimModel
from RatEnv.RL_Controller import MouseController

SceneFile = "../models/scene_test1.xml"
# SceneFile = "../models/dynamic_4l_t3.xml"

# RUN_STEPS = 10000
RUN_STEPS = 5000


class DATA_Recorder():
    def __init__(self):
        self.imu_pos = deque([])
        self.imu_quat = deque([])
        self.imu_vel = deque([])
        self.imu_acc = deque([])
        self.imu_gyro = deque([])
        self.curstep = deque([])

    def update(self, theMouse: SimModel, theController: MouseController):
        self.imu_pos.append(theMouse.pos)
        self.imu_quat.append(theMouse.quat)
        self.imu_vel.append(theMouse.vel)
        self.imu_acc.append(theMouse.acc)
        self.imu_gyro.append(theMouse.gyro)

        self.curstep.append((theController.curStep+theController.SteNum-1)%theController.SteNum)  # Pre


    def savePath2(self, FileName):
        # scio.savemat('mvpath.mat', {'H_range': self.movePath})  # 写入mat文件
        imu_pos = np.array(self.imu_pos)
        imu_quat = np.array(self.imu_quat)
        imu_vel = np.array(self.imu_vel)
        imu_acc = np.array(self.imu_acc)
        imu_gyro = np.array(self.imu_gyro)
        imu_curstep = self.curstep
        scio.savemat(FileName + '.mat', {'pos': imu_pos, 'quat': imu_quat, 'vel': imu_vel,
                                         'acc': imu_acc, 'gyro': imu_gyro, 'curstep': imu_curstep})  # 写入mat文件



if __name__ == '__main__':
    env = RatRL(SceneFile, Render=False)
    Record = DATA_Recorder()
    R = []
    V_vels = []
    Delta_vels = []
    action = [1., 1., 1., 1.]

    s = env.reset()

    for i in range(RUN_STEPS):

        # From RL_wrapper3_Connect V0.9
        ActionSignal = action

        tCtrlData = env.theController.runStep(ActionSignal)  # No Spine
        env.theMouse.runStep(tCtrlData)
        # env.render()

        Record.update(env.theMouse, env.theController)

    Record.savePath2("imu_record_Scene1_0125")

    gyros = np.array(Record.imu_gyro).transpose()

    plot_simple([gyros[0], gyros[1], gyros[2]])