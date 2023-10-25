import numpy as np
import math

from RatEnv.LegModel.RL_forPath import LegPath2
# -----------------------------------------------------------
from RatEnv.LegModel.foreLeg import ForeLegM
from RatEnv.LegModel.hindLeg import HindLegM


class MouseController(object):
    """docstring for MouseController"""

    def __init__(self, SteNum=376):
        super(MouseController, self).__init__()
        PI = np.pi
        self.curStep = 0  # Spine

        self.turn_F = -2 * PI / 180  # -2 * PI / 180 ---some = 0
        self.turn_H = 5 * PI / 180  # 5 * PI / 180 --- some = 0
        self.pathStore = LegPath2()
        # [LF, RF, LH, RH]
        # --------------------------------------------------------------------- #
        # self.phaseDiff = [0, PI, PI*1/2, PI*3/2]	# Walk
        # self.period = 3/2
        # self.SteNum = 36							#32 # Devide 2*PI to multiple steps
        # self.spinePhase = self.phaseDiff[3]
        # --------------------------------------------------------------------- #
        self.phaseDiff = [0, PI, PI, 0]  # Trot
        self.period = 2 / 2
        self.SteNum = SteNum
        # print("SteNum ----> ", self.SteNum)
        self.spinePhase = self.phaseDiff[2] - PI  # theta_spine=0, when theta_hl = pi
        # --------------------------------------------------------------------- #
        self.spine_A = 0  # 10 a_s = 2theta_s
        # print("angle --> ", self.spine_A)
        self.spine_A = self.spine_A * PI / 180
        # --------------------------------------------------------------------- #
        fl_params = {'lr0': 0.033, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0295,
                     'l2': 0.0145, 'l3': 0.0225, 'l4': 0.0145, 'alpha': 23 * np.pi / 180}
        self.fl_left = ForeLegM(fl_params)
        self.fl_right = ForeLegM(fl_params)
        # --------------------------------------------------------------------- #
        hl_params = {'lr0': 0.032, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0317,
                     'l2': 0.02, 'l3': 0.0305, 'l4': 0.0205, 'alpha': 73 * np.pi / 180}
        self.hl_left = HindLegM(hl_params)
        self.hl_right = HindLegM(hl_params)
        # --------------------------------------------------------------------- #
        self.stepDiff = [0, 0, 0, 0]
        for i in range(4):
            self.stepDiff[i] = int(self.SteNum * self.phaseDiff[i] / (2 * PI))
        self.stepDiff.append(int(self.SteNum * self.spinePhase / (2 * PI)))

        # 3_1 piece 4
        # Act_Per = int(self.SteNum / 4)
        # self.Action_Div = [Act_Per, Act_Per, Act_Per, self.SteNum - Act_Per*3]  # 93, 93, 93, 94
        # 3_2 piece 8
        # self.Action_Div = [47, 47, 47, 47, 47, 47, 47, 47]  # 93, 93, 93, 94
        # TODO: Move to Wrapper3_Connect Done

        # Stores and Variables
        self.trgXList = [[], [], [], []]
        self.trgYList = [[], [], [], []]
        self.curStep = 0  # Spine

    def reset(self):
        self.trgXList = [[], [], [], []]
        self.trgYList = [[], [], [], []]
        self.curStep = 0  # Spine

    def getLegCtrl(self, leg_M, curStep, leg_ID, ActionSignal):
        curStep = curStep % self.SteNum
        turnAngle = self.turn_F
        leg_flag = "F"
        if leg_ID > 1:
            leg_flag = "H"
            turnAngle = self.turn_H
        theta = 2 * np.pi * curStep / self.SteNum

        # RL_forPath
        # currentrho = self.pathStore.getinirho(theta, leg_flag, ActionSignal)
        # # currentPos = self.pathStore.getOvalPathPoint(currentrho, theta, leg_flag, self.period)
        # currentPos = self.pathStore.getOvalPoint(theta, currentrho, leg_flag)

        # RL_forPath---LegPath2
        currentPos = self.pathStore.getOvalPathPoint(theta, leg_flag, ActionSignal)

        trg_x = currentPos[0]
        trg_y = currentPos[1]

        self.trgXList[leg_ID].append(trg_x)
        self.trgYList[leg_ID].append(trg_y)

        tX = math.cos(turnAngle) * trg_x - math.sin(turnAngle) * trg_y  # 进行倾角纠正，得到准确的末端点
        tY = math.cos(turnAngle) * trg_y + math.sin(turnAngle) * trg_x
        qVal = leg_M.pos_2_angle(tX, tY)  # get q

        # for q in qVal:
        #     if q is None or q <-1.57 or q > 1.57:
        #         print("ERRORQ [curstep, Action]=[{}, {}]--leg={}".format(curStep, ActionSignal, leg_flag))
        #         print(qVal)
        #         break

        return qVal

    def getSpineVal(self, spinestep):
        radian = 2 * np.pi * spinestep / self.SteNum
        return self.spine_A * math.cos(radian)  # 0-->1.0, pi-->-1

    # spinePhase = 2*np.pi*spineStep/self.SteNum
    # return self.spine_A*math.sin(spinePhase)

    def runStep(self, action):
        foreLeg_left_q = self.getLegCtrl(self.fl_left,
                                         self.curStep + self.stepDiff[0], 0, action[0])
        foreLeg_right_q = self.getLegCtrl(self.fl_right,
                                          self.curStep + self.stepDiff[1], 1, action[1])
        hindLeg_left_q = self.getLegCtrl(self.hl_left,
                                         self.curStep + self.stepDiff[2], 2, action[2])
        hindLeg_right_q = self.getLegCtrl(self.hl_right,
                                          self.curStep + self.stepDiff[3], 3, action[3])

        spineStep = (self.curStep + self.stepDiff[4]) % self.SteNum
        if self.spine_A:
            spine = self.getSpineVal(spineStep)
            tail = -spine * np.pi / self.spine_A
        else:
            spine = 0
            tail = 0
        self.curStep = (self.curStep + 1) % self.SteNum

        ctrlData = []

        # foreLeg_left_q = [1,0]
        # foreLeg_right_q = [1,0]
        # hindLeg_left_q = [-1,0]
        # hindLeg_right_q = [-1,0]
        ctrlData.extend(foreLeg_left_q)
        ctrlData.extend(foreLeg_right_q)
        ctrlData.extend(hindLeg_left_q)
        ctrlData.extend(hindLeg_right_q)
        ctrlData.extend([tail, 0, 0, spine])
        return ctrlData
