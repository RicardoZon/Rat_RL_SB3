import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math
import scipy.io as scio

from RatEnv.LegModel.foreLeg import ForeLegM
from RatEnv.LegModel.hindLeg import HindLegM




if __name__ == '__main__':
    fl_params = {'lr0': 0.033, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0295,
                 'l2': 0.0145, 'l3': 0.0225, 'l4': 0.0145, 'alpha': 23 * np.pi / 180}
    hl_params = {'lr0': 0.032, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0317,
                 'l2': 0.02, 'l3': 0.0305, 'l4': 0.0205, 'alpha': 73 * np.pi / 180}

    m1_range = {"max": 1.57,
                "min": -1.57}
    m2_range = {"max": 1.57,
                "min": -2.01}  # 2.01 ~ 120 Deg

    # legmodel = ForeLegM(fl_params)
    legmodel = HindLegM(hl_params)

    Y_range = np.arange(-0.08, 0.08, 0.0001)
    Z_range = np.arange(-0.08, 0.06, 0.0001)

    H_range = deque([])
    for tY in Y_range:
        for tZ in Z_range:
            [q1, q2] = legmodel.pos_2_angle(tY, tZ)
            if q1==None or q2==None:
                continue
            if q1 < m1_range["max"] and q1 > m1_range["min"] and q2 < m2_range["max"] and q2 > m2_range["min"]:
                H_range.append([tY, tZ])

    H_range = np.array(H_range)
    # scio.savemat('rangesearch_fl_kineYH.mat', {'H_range': H_range})
    scio.savemat('rangesearch_hl_kineYH.mat', {'H_range': H_range})
