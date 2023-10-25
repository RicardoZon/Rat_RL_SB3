import numpy as np
from RatEnv.LegModel.bezier import bezier
import math

class LegPath_Bezier():
    def __init__(self):
        self.Centers=[
            [0.000, -0.050], [0.000, -0.050],  # FL, FR  # -0.050~-0.060
            [0.00, -0.065], [0.00, -0.065]  # HL, HR  # -0.055~ -0.065
        ]
        self.vs = []
        self.H_stance_m = [0.010, 0.010, 0.020, 0.020]
        # self.H_stance_m

    def getOvalPathPoint(self, phase, leg_ID: int):
        Center = self.Centers[leg_ID]

        H_stance_m = 2 / 1000.0  # 0.005
        H_swing_m = self.H_stance_m[leg_ID] #  = 15 / 1000.0  # 0.040
        Vx_mps = 80 / 1000.0 + 0.001

        T_stride_s = 1000 / 1000.0  # 1s
        overlay = 0 / 100.0
        T_stance_s = T_stride_s * (0.5 + overlay)  # 0.200
        self.T_swing_s = T_stride_s * (0.5 - overlay)  # 0.200
        Z0 = 0 / 1000.0  # standing height  130
        x_begin = -Vx_mps * (0.5) * T_stance_s
        x_end = Vx_mps * (0.5) * T_stance_s

        n = 12  # point count in x_points
        A = Vx_mps * self.T_swing_s / (n - 1)
        B = 2.00 * A

        # XZ Bezier curve
        xz_points = np.array(
            [
                [x_begin, 0.0, 0.0],  # P0
                [x_begin - A, 0.0, 0.0],  # P1
                [x_begin - B, 0.0, H_swing_m],  # P2
                [x_begin - B, 0.0, H_swing_m],  # P3
                [x_begin - B, 0.0, H_swing_m],  # P4
                [0.0, 0.0, H_swing_m],  # P5
                [0.0, 0.0, H_swing_m],  # P6
                [0.0, 0.0, H_swing_m * 1.2],  # P7
                [x_end + B, 0.0, H_swing_m * 1.2],  # P8
                [x_end + B, 0.0, H_swing_m * 1.2],  # P9
                [x_end + A, 0.0, 0.0],  # P10
                [x_end, 0.0, 0.0]  # P11
            ]
        ).transpose()

        # [0-1] swing, (1-2) stance
        phase = phase % 2
        if phase > 1:  # Stance
            multiplier = phase - 1
            x = -Vx_mps * (multiplier - 0.5) * T_stance_s
            z = Z0 - H_stance_m * math.cos(math.pi * (multiplier - 0.5))
            x = Center[0] - x  # For Direction
            z = Center[1] + z

        else:  # Swing [0, 1]
            multiplier = phase
            p = bezier(multiplier, xz_points)
            x = p[0, 0]
            z = p[2, 0]
            z = z + Z0
            x = Center[0] - x  # For Direction
            z = Center[1] + z
        return [x, z]

    def cal(self):
        H_stance_m = 2 / 1000.0  # 0.005
        H_swing_m = 15 / 1000.0  # 0.040
        Vx_mps = 80 / 1000.0 + 0.001

        T_stride_s = 400/1000.0
        overlay = 0/100.0
        T_stance_s = T_stride_s * (0.5+overlay) # 0.200
        self.T_swing_s = T_stride_s * (0.5-overlay) # 0.200
        Z0 = 0/1000.0  # standing height  130
        x_begin = -Vx_mps * (0.5) * T_stance_s
        x_end   =  Vx_mps * (0.5) * T_stance_s

        n = 12  # point count in x_points
        A = Vx_mps * self.T_swing_s / (n - 1)
        B = 2.00 * A

        # XZ Bezier curve
        xz_points = np.array(
            [
                [x_begin,     0.0, 0.0],  # P0
                [x_begin - A, 0.0, 0.0],  # P1
                [x_begin - B, 0.0, H_swing_m],  # P2
                [x_begin - B, 0.0, H_swing_m],  # P3
                [x_begin - B, 0.0, H_swing_m],  # P4
                [0.0,         0.0, H_swing_m],  # P5
                [0.0,         0.0, H_swing_m],  # P6
                [0.0,         0.0, H_swing_m * 1.2],  # P7
                [x_end + B,   0.0, H_swing_m * 1.2],  # P8
                [x_end + B,   0.0, H_swing_m * 1.2],  # P9
                [x_end + A,   0.0, 0.0],  # P10
                [x_end,       0.0, 0.0]  # P11
            ]
        ).transpose()

        # build data
        step_s = 0.0005  # s  ~ 0.5ms

        # .. for stride XZ position
        self.x_dataset = []
        self.z_dataset = []
        x_min = 1.000
        x_max = -1.000
        z_min = 1.000
        z_max = -1.000

        for t in np.arange(step_s, T_stance_s, step_s):
            multiplier = t / T_stance_s
            x = -Vx_mps * (multiplier - 0.5) * T_stance_s
            z = Z0 - H_stance_m * math.cos(math.pi * (multiplier - 0.5))

            # # circle test
            # if self.variables["Curve"].get() == "Circle":
            #     r = 0.04
            #     x = r * math.cos(math.pi * 2 * t / (T_stance_s + self.T_swing_s))
            #     z = Z0 + r * math.sin(math.pi * 2 * t / (T_stance_s + self.T_swing_s))

            x = self.para_CF[0] - x  # For Direction
            z = self.para_CF[1] + z
            self.x_dataset.append(x)
            self.z_dataset.append(z)
            x_max = max(x, x_max)
            x_min = min(x, x_min)
            z_max = max(z, z_max)
            z_min = min(z, z_min)

        for t in np.arange(0, self.T_swing_s + step_s, step_s):
            multiplier = t / self.T_swing_s
            print(multiplier)
            x = 0
            z = 0

            p = bezier(multiplier, xz_points)
            x = p[0, 0]
            z = p[2, 0]

            z = z + Z0

            x = self.para_CF[0] - x  # For Direction
            z = self.para_CF[1] + z
            self.x_dataset.append(x)
            self.z_dataset.append(z)
            x_max = max(x, x_max)
            x_min = min(x, x_min)
            z_max = max(z, z_max)
            z_min = min(z, z_min)

        self.x_dataset.append(self.x_dataset[0])
        self.z_dataset.append(self.z_dataset[0])



if __name__ == "__main__":
    generator = LegPath_Bezier()
    # generator.cal()
    x_dataset = []
    z_dataset = []
    for p in np.arange(0, 2, 0.001):
        [x, z] = generator.getOvalPathPoint(p, leg_ID=0)
        x_dataset.append(x)
        z_dataset.append(z)

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    # ax.plot(generator.x_dataset, generator.z_dataset)
    ax.plot(x_dataset, z_dataset)

    ax.axis('equal')
    fig.show()
