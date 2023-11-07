from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import deque
import scipy.io as scio
import time
from RatEnv.Controller import MouseController
import argparse
from mujoco_py.generated import const
import torch
from ToSim import SimModel
import pygame
import sys

from RatEnv.LegModel.forPath import LegPath
from RatEnv.LegModel.foreLeg import ForeLegM
from RatEnv.LegModel.hindLeg import HindLegM


class MotionPlanner(object):
    """docstring for MouseController"""

    def __init__(self, SteNum=376):
        super(MotionPlanner, self).__init__()
        PI = np.pi
        self.curStep = 0  # Spine

        self.turn_F = -2 * PI / 180
        self.turn_H = 5 * PI / 180
        self.pathStore = LegPath()
        self.phaseDiff = [0, PI, PI, 0]  # Trot
        self.period = 2 / 2
        self.SteNum = SteNum
        print("----> ", self.SteNum)
        self.spinePhase = self.phaseDiff[2] - PI  # theta_spine=0, when theta_hl = pi
        # --------------------------------------------------------------------- #
        self.spine_A = 0  # 10 a_s = 2theta_s
        print("angle --> ", self.spine_A)
        self.spine_A = self.spine_A * PI / 180
        # --------------------------------------------------------------------- #
        fl_params = {'lr0': 0.033, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0295,
                     'l2': 0.0145, 'l3': 0.0225, 'l4': 0.0145, 'alpha': 23 * np.pi / 180}
        self._fl_left = ForeLegM(fl_params)
        self._fl_right = ForeLegM(fl_params)
        # --------------------------------------------------------------------- #
        hl_params = {'lr0': 0.032, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0317,
                     'l2': 0.02, 'l3': 0.0305, 'l4': 0.0205, 'alpha': 73 * np.pi / 180}
        self._hl_left = HindLegM(hl_params)
        self._hl_right = HindLegM(hl_params)
        # --------------------------------------------------------------------- #
        self.trgXList = [[], [], [], []]
        self.trgYList = [[], [], [], []]
        self.leg_models = [self._fl_left, self._fl_right, self._hl_left, self._hl_right]

        self.spine_rad = 0.0

    def getLegCtrl(self, leg_M, curStep, leg_ID):
        curStep = curStep % self.SteNum
        turnAngle = self.turn_F
        leg_flag = "F"
        if leg_ID > 1:
            leg_flag = "H"
            turnAngle = self.turn_H
        radian = 2 * np.pi * curStep / self.SteNum
        currentPos = self.pathStore.getOvalPathPoint(radian, leg_flag, self.period)
        trg_x = currentPos[0]
        trg_y = currentPos[1]
        self.trgXList[leg_ID].append(trg_x)
        self.trgYList[leg_ID].append(trg_y)

        tX = math.cos(turnAngle) * trg_x - math.sin(turnAngle) * trg_y
        tY = math.cos(turnAngle) * trg_y + math.sin(turnAngle) * trg_x
        qVal = leg_M.pos_2_angle(tX, tY)
        return qVal

    def getSpineVal(self, spinestep):
        radian = 2 * np.pi * spinestep / self.SteNum
        return self.spine_A * math.cos(radian)  # 0-->1.0, pi-->-1

    # spinePhase = 2*np.pi*spineStep/self.SteNum
    # return self.spine_A*math.sin(spinePhase)

    def Base(self, Env: SimModel):
        ctrlData = [0.0, 1.2, 0.0, 1.2, 0.0, -1.2, 0.0,-1.2, 0,0,0,0]
        Planner.spine_rad = ctrlData[11]
        self.LinearSmooth(Env, ctrlData)
        # for _ in range(int(self.SteNum / 2)):
        #     Env.runStep(ctrlData, render=True)


    def Spine(self, Env: SimModel, spine=0.):
        self.spine_rad = spine
        ctrlData = theMouse.sim.data.ctrl.copy()
        ctrlData[11] = spine  # Spine
        self.LinearSmooth(Env, ctrlData)

    def LegUP(self, Env: SimModel, leg_ID):
        # 0,1,2,3 --> LF, RF, LH, RH
        ctrlData = theMouse.sim.data.ctrl.copy()
        for curStep in range(int(self.SteNum/2)):
            qs = self.getLegCtrl(self.leg_models[leg_ID], curStep, leg_ID)
            qs[1] = qs[1] + 0.2
            ctrlData[leg_ID*2: leg_ID*2+2] = qs
            Env.runStep(ctrlData, render=True)

    def LegDOWN(self, Env: SimModel, leg_ID):
        # 0,1,2,3 --> LF, RF, LH, RH
        ctrlData = theMouse.sim.data.ctrl.copy()
        for curStep in range(int(self.SteNum/2), self.SteNum):
            qs = self.getLegCtrl(self.leg_models[leg_ID], curStep, leg_ID)
            ctrlData[leg_ID*2: leg_ID*2+2] = qs
            Env.runStep(ctrlData, render=True)

    def Paw(self, Env: SimModel, delta=0.2, nums=[0, 2, 4, 6]):
        # 0,1,2,3 --> LF, RF, LH, RH
        ctrlData = theMouse.sim.data.ctrl.copy()
        for index in nums:
            ctrlData[index] = ctrlData[index] + delta
        self.LinearSmooth(Env, ctrlData)

    def FrontPawUp(self, Env: SimModel, leg_ID: int):
        ctrlData = theMouse.sim.data.ctrl.copy()
        ctrlData[leg_ID*2+1] = 0.0
        self.LinearSmooth(Env, ctrlData) # Up

        ctrlData[leg_ID*2] = -0.7
        self.LinearSmooth(Env, ctrlData)  # Up

        ctrlData[leg_ID * 2 + 1] = 1.1
        self.LinearSmooth(Env, ctrlData)  # U

    def FrontPawDown(self, Env: SimModel, leg_ID: int):
        ctrlData = theMouse.sim.data.ctrl.copy()
        ctrlData[leg_ID*2:leg_ID*2 + 2] = [0.5, 1.1]
        self.LinearSmooth(Env, ctrlData,  Len=60)  # Up

    def LinearSmooth(self, Env: SimModel, ctrlData_target: list, Len=60):
        ctrlData = theMouse.sim.data.ctrl.copy()  # array
        SteNum = Len
        deltas = -(ctrlData - ctrlData_target)/SteNum
        for _ in range(SteNum):
            ctrlData = ctrlData + deltas
            Env.runStep(ctrlData, render=True)
        print(ctrlData)
        print(ctrlData_target)


    def runStep(self, dir=1):
        foreLeg_left_q = self.getLegCtrl(self.fl_left,
                                         self.curStep + self.stepDiff[0], 0)
        foreLeg_right_q = self.getLegCtrl(self.fl_right,
                                          self.curStep + self.stepDiff[1], 1)
        hindLeg_left_q = self.getLegCtrl(self.hl_left,
                                         self.curStep + self.stepDiff[2], 2)
        hindLeg_right_q = self.getLegCtrl(self.hl_right,
                                          self.curStep + self.stepDiff[3], 3)

        spineStep = (self.curStep + self.stepDiff[4]) % self.SteNum
        if self.spine_A:
            spine = self.getSpineVal(spineStep)
            tail = -spine * np.pi / self.spine_A
        else:
            spine = 0
            tail = 0

        if dir == 1:
            self.curStep = (self.curStep + 1) % self.SteNum
        else:
            self.curStep = (self.curStep + self.SteNum - 1) % self.SteNum

        ctrlData = []
        ctrlData.extend(foreLeg_left_q)
        ctrlData.extend(foreLeg_right_q)
        ctrlData.extend(hindLeg_left_q)
        ctrlData.extend(hindLeg_right_q)
        ctrlData.extend([tail, 0, 0, spine])
        return ctrlData




if __name__ == '__main__':
    # state = torch.load('../Outputs/Trapped_Scenario4_Stairs_Sparse.pth')
    state = None
    RENDER = True
    # MODELPATH = "../models/dynamic_4l_t3.xml"
    # MODELPATH = "../models/Scenario1_Planks.xml"
    MODELPATH = "../models/Scenario4_Stairs_Sparse.xml"
    RUN_STEPS = 10000

    pygame.init()
    display = pygame.display.set_mode((300, 300))

    theMouse = SimModel(MODELPATH, Render=RENDER)
    if not state:
        state = theMouse.sim_state
    frame_skip = 1
    dt = theMouse.model.opt.timestep*frame_skip
    fre_cyc = 0.67  # 1.25  # 0.80?
    # SteNum = int(1 / (dt * fre_cyc) / 2)  # /1.25)
    SteNum = 376

    theController = MouseController(SteNum=SteNum)
    Planner = MotionPlanner(SteNum=SteNum)

    for i in range(500):
        ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0,-1.2, 0,0,0,0]
        theMouse.runStep(ctrlData)
    theMouse.initializing()
    start = time.time()

    # theMouse.sim.set_state(state)
    def run_tmp(dir=1, steps=RUN_STEPS, spine=0.):
        for i in range(steps):
            pos_pre = theMouse.pos.copy()
            ctrlData = theController.runStep(dir=dir)				# No Spine
            # ctrlData[6] = -1.57
            # ctrlData[7] = 1.57  # Work
            ctrlData[11] = spine

            for _ in range(frame_skip):
                theMouse.runStep(ctrlData, render=RENDER)
            pos = theMouse.pos

            v = (pos[1]-pos_pre[1])*(-4)/dt
    run_tmp(dir=1, steps=SteNum*3, spine=0.)

    while 1:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:  # Reset
                    theMouse.sim.set_state(state)
                    run_tmp(dir=1, steps=SteNum * 3, spine=Planner.spine_rad)
                # Default Trot Controller
                if event.key == pygame.K_i:  # Trot ahead
                    run_tmp(dir=1, steps=SteNum, spine=Planner.spine_rad)
                if event.key == pygame.K_k:  # Trot back
                    run_tmp(dir=0, steps=SteNum, spine=Planner.spine_rad)

                # Stand Base
                if event.key == pygame.K_0:  # Base
                    Planner.Base(theMouse)

                # Spine --------------------------------------
                if event.key == pygame.K_7:  # Spine --> left
                    Planner.Spine(theMouse, spine=1.5)
                if event.key == pygame.K_9:  # Spine --> right
                    Planner.Spine(theMouse, spine=-1.5)
                if event.key == pygame.K_8:  # Spine --> Zero
                    Planner.Spine(theMouse, spine=0.0)

                # Singel Leg Motion: Hind Leg --------------------------
                if event.key == pygame.K_a:  # Pick Up LH
                    Planner.LegUP(theMouse, 2)
                if event.key == pygame.K_s:  # Pick Up RH
                    Planner.LegUP(theMouse, 3)
                if event.key == pygame.K_z:  # Down LH
                    Planner.LegDOWN(theMouse, 2)
                if event.key == pygame.K_x:  # Down RH
                    Planner.LegDOWN(theMouse, 3)
                # Single Leg Motion: Front Leg --------------------------
                if event.key == pygame.K_1:  # Pick Up LF
                    Planner.LegUP(theMouse, 0)
                if event.key == pygame.K_2:  # Pick Up RF
                    Planner.LegUP(theMouse, 1)
                if event.key == pygame.K_q:  # Down LF
                    Planner.LegDOWN(theMouse, 0)
                if event.key == pygame.K_w:  # Down RF
                    Planner.LegDOWN(theMouse, 1)

                if event.key == pygame.K_3:  # UP LF
                    Planner.Paw(theMouse, delta=-0.2, nums=[0])
                if event.key == pygame.K_e:  # Down LF
                    Planner.Paw(theMouse, delta=0.2, nums=[0])
                if event.key == pygame.K_4:  #
                    Planner.Paw(theMouse, delta=-0.2, nums=[1])
                if event.key == pygame.K_r:  #
                    Planner.Paw(theMouse, delta=0.2, nums=[1])

                if event.key == pygame.K_5:  # Down RF
                    Planner.Paw(theMouse, delta=-0.2, nums=[2])
                if event.key == pygame.K_t:  # Down RF
                    Planner.Paw(theMouse, delta=0.1, nums=[2])
                if event.key == pygame.K_6:  #
                    Planner.Paw(theMouse, delta=-0.2, nums=[3])
                if event.key == pygame.K_y:  #
                    Planner.Paw(theMouse, delta=0.2, nums=[3])

                if event.key == pygame.K_d:  # UP LF
                    Planner.Paw(theMouse, delta=-0.2, nums=[4])
                if event.key == pygame.K_c:  # Down LF
                    Planner.Paw(theMouse, delta=0.2, nums=[4])
                if event.key == pygame.K_f:  #
                    Planner.Paw(theMouse, delta=0.2, nums=[5])
                if event.key == pygame.K_v:  #
                    Planner.Paw(theMouse, delta=-0.2, nums=[5])

                if event.key == pygame.K_g:  # UP LH
                    Planner.Paw(theMouse, delta=-0.2, nums=[6])
                if event.key == pygame.K_b:  # Down LH
                    Planner.Paw(theMouse, delta=0.2, nums=[6])
                if event.key == pygame.K_h:  #
                    Planner.Paw(theMouse, delta=0.2, nums=[7])
                if event.key == pygame.K_n:  #
                    Planner.Paw(theMouse, delta=-0.2, nums=[7])

                if event.key == pygame.K_u:  #
                    Planner.FrontPawUp(theMouse, leg_ID=0)
                if event.key == pygame.K_o:  #
                    Planner.FrontPawUp(theMouse, leg_ID=1)
                if event.key == pygame.K_j:  #
                    Planner.FrontPawDown(theMouse, leg_ID=0)
                if event.key == pygame.K_l:  #
                    Planner.FrontPawDown(theMouse, leg_ID=1)



    # run_tmp(dir=0, steps=700)
    run_tmp(dir=1, steps=RUN_STEPS)
    end = time.time()
    timeCost = end-start
    print("Time -> ", timeCost)
    # print(pos)

    '''
    s_trap = theMouse.sim.get_state()
    torch.save(s_trap, './XXX.pth')
    '''
