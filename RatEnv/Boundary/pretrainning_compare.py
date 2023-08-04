import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import scipy.io as scio
from fit_boundary import FitNet as BoundaryNet
from fit_boundary import cart2pol
from fit_boundary import Sdraw

def plangenerate(RadiusU, RadiusD):
    theta1 = np.arange(0, np.pi, 0.005)
    theta2 = np.arange(np.pi, 2*np.pi, 0.005)

    X = np.append(RadiusU[0] * np.cos(theta1), RadiusD[0] * np.cos(theta2))
    Y = np.append(RadiusU[1] * np.sin(theta1), RadiusD[1] * np.sin(theta2))
    Sdraw(X, Y)
    theta, rho = cart2pol(X, Y)
    theta = theta[:, None]  # [Len, 1]
    rho = rho[:, None]  # [Len, 1]
    Sdraw(theta, rho)
    return theta, rho

def computeratio(BNet, theta, rho, device):
    thetaT = torch.as_tensor(theta, device=device)
    rhomax = BNet(thetaT).detach().cpu().numpy()
    rhomax = rhomax * 0.98  # Correct
    ratio = rho / rhomax  # 得到rho和rhomax之间的比例
    return ratio


def test(BNet, theta, ratio, device):
    thetaT = torch.as_tensor(theta, device=device)
    rhomax = BNet(thetaT).detach().cpu().numpy()
    rhomax = rhomax * 0.98  # Correct

    rho = rhomax * ratio
    Sdraw(theta, rho)


if __name__ == '__main__':
    FU = [[-0.00, -0.045], [0.03, 0.01]]
    FD = [[-0.00, -0.045], [0.03, 0.005]]
    HU = [[0.00, -0.05], [0.03, 0.01]]
    HD = [[0.00, -0.05], [0.03, 0.005]]
    device = "cpu"

    BNetF = BoundaryNet().to(device).double()
    BNetF.load_state_dict(torch.load('front_boundary.pth'))
    thetaf, rhof = plangenerate(FU[1], FD[1])  # 根据椭圆参数生成的，自定义的运动曲线
    ratiof = computeratio(BNetF, thetaf, rhof, device)

    BNetH = BoundaryNet().to(device).double()
    BNetH.load_state_dict(torch.load('hind_boundary.pth'))
    thetah, rhoh = plangenerate(HU[1], HD[1])
    ratioh = computeratio(BNetH, thetah, rhoh, device)

    # test(BNetF, thetaf, ratiof, device)
    # test(BNetH, thetah, ratioh, device)

    # scio.savemat('ActionPre_front_kineYH.mat', {'theta': thetaf, 'ratio': ratiof})
    # scio.savemat('ActionPre_hind_kineYH.mat', {'theta': thetah, 'ratio': ratioh})
