import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math
import scipy.io as scio

from RatEnv.LegModel.foreLeg import ForeLegM
from RatEnv.LegModel.hindLeg import HindLegM
from RatEnv.Controller import MouseController
from RatEnv.Controller import MouseControllerB
from RatEnv.LegModel.forPath_Bezier import LegPath_Bezier

def Search(legmodel):
    Y_range = np.arange(-0.08, 0.08, 0.001)
    Z_range = np.arange(-0.08, 0.06, 0.001)
    H_range = deque([])
    for tY in Y_range:
        for tZ in Z_range:
            [q1, q2] = legmodel.pos_2_angle(tY, tZ)
            if q1 == None or q2 == None:
                continue
            if q1 < m1_range["max"] and q1 > m1_range["min"] and q2 < m2_range["max"] and q2 > m2_range["min"]:
                H_range.append([tY, tZ])

    H_range = np.array(H_range)
    return H_range

if __name__ == '__main__':
    fl_params = {'lr0': 0.033, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0295,
                 'l2': 0.0145, 'l3': 0.0225, 'l4': 0.0145, 'alpha': 23 * np.pi / 180}
    hl_params = {'lr0': 0.032, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0317,
                 'l2': 0.02, 'l3': 0.0305, 'l4': 0.0205, 'alpha': 73 * np.pi / 180}

    m1_range = {"max": 1.57,
                "min": -1.57}
    m2_range = {"max": 1.57,
                "min": -1.57}  # 2.01 ~ 120 Deg

    turn_H = 5 * np.pi / 180

    legmodel_FL = ForeLegM(fl_params)
    # legmodel_FL = HindLegM(hl_params)

    H_range_FL = Search(legmodel_FL)
    theController = MouseController(SteNum=376)
    theController.pathStore.para_FU = [[-0.005, -0.045], [0.03, 0.01]]
    theController.pathStore.para_FD = [[-0.005, -0.045], [0.03, 0.005]]
    # theController.pathStore.para_HU = [[0.005, -0.055], [0.03, 0.01]]
    # theController.pathStore.para_HD = [[0.005, -0.055], [0.03, 0.005]]
    # theController.pathBezier.Centers[0] = [0.000, -0.060]  # -0.050~-0.060
    # theController.pathBezier.Centers[2] = [0.00, -0.065]  # -0.055~ -0.065
    for _ in range(theController.SteNum):
        _ = theController.runStep()  # No Spine


    # Bezier Simple
    generator = LegPath_Bezier()
    x_dataset = []
    z_dataset = []
    for p in np.arange(0, 2, 0.001):
        [x, z] = generator.getOvalPathPoint(p, leg_ID=0)
        x_dataset.append(x)
        z_dataset.append(z)

    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(H_range_FL[:, 0], H_range_FL[:, 1])
    ax.plot(theController.trgXList[0], theController.trgYList[0], 'r')
    # ax.plot(x_dataset, z_dataset, 'r')
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    # ax.scatter(Poses[:, 0], Poses[:, 1])
    ax.set_aspect("equal")
    fig.show()

    # scio.savemat('rangesearch_fl_kineYH.mat', {'H_range': H_range})
    # scio.savemat('rangesearch_hl_kineYH.mat', {'H_range': H_range})

    legmodel_HL = HindLegM(hl_params)
    H_range_HL = Search(legmodel_HL)
    generator = LegPath_Bezier()
    x_dataset = []
    z_dataset = []
    for p in np.arange(0, 2, 0.001):
        [x, z] = generator.getOvalPathPoint(p, leg_ID=2)
        x_dataset.append(x)
        z_dataset.append(z)

    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(H_range_HL[:, 0], H_range_HL[:, 1])
    ax.plot(theController.trgXList[2], theController.trgYList[2], 'r')
    # ax.plot(x_dataset, z_dataset, 'r')
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    # ax.scatter(Poses[:, 0], Poses[:, 1])
    ax.set_aspect("equal")
    fig.show()