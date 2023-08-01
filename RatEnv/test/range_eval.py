import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math
import scipy.io as scio

from RatEnv.RL_Controller import MouseController


if __name__ == '__main__':
    theController = MouseController(0.67)

    Signal_range = np.arange(0., 1.0001, 0.0001)

    # thetas = np.arange(0, 2 * np.pi, 0.001)
    # leg_flags = ["F", "H"]
    # for leg_flag in leg_flags:
    #     for theta in thetas:
    #         for action in Signal_range:
    #             currentPos = theController.pathStore.getOvalPathPoint(theta, leg_flag, action)
    #             if currentPos[0] == None or currentPos[1] == None:
    #                 print(currentPos)

    steps = np.arange(0, theController.SteNum)
    for curstep in steps:
        for action in Signal_range:
            foreLeg_left_q = theController.getLegCtrl(theController.fl_left,
                                             curstep, 0, action)
            hindLeg_left_q = theController.getLegCtrl(theController.hl_left,
                                             curstep, 2, action)


    print("Complete")
