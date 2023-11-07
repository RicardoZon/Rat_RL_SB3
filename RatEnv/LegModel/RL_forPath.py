import math
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
# Boundary

class BoundaryNet(nn.Module):
    def __init__(self):
        super(BoundaryNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class LegPath(object):
	"""docstring for ForeLegPath"""
	def __init__(self, pathType="circle", device="cpu"):
		super(LegPath, self).__init__()
		# Trot
		self.para_CF = [-0.00, -0.045]
		self.para_CH = [-0.00, -0.050]

		self.para_FU = [[-0.00, -0.045], [0.03, 0.01]]
		self.para_FD = [[-0.00, -0.045], [0.03, 0.005]]
		self.para_HU = [[0.00, -0.05], [0.03, 0.01]]
		self.para_HD = [[0.00, -0.05], [0.03, 0.005]]

		self.device = device
		self.BoundaryF = BoundaryNet().to(device).double()
		self.BoundaryF.load_state_dict(torch.load('../RatEnv/fl_boundary.pth'))
		self.BoundaryF.eval()
		self.BoundaryH = BoundaryNet().to(device).double()
		self.BoundaryH.load_state_dict(torch.load('../RatEnv/hl_boundary.pth'))
		self.BoundaryH.eval()

		# self.test()

	def test(self):
		theta = np.arange(-np.pi, np.pi, 0.005)
		thetaT = torch.tensor(theta, dtype=torch.float64, device=self.device)
		thetaT = thetaT.unsqueeze(1)  # [BS 1]
		rho_Train = self.BoundaryF(thetaT).detach().cpu().numpy()
		plt.figure()
		plt.plot(theta, rho_Train)
		plt.show()

	def getinirho(self, theta, leg_flag, ActionSignal):
		theta = (theta + np.pi)% (2 * np.pi) - np.pi
		thetaT = torch.tensor(theta, dtype=torch.float64, device=self.device)
		thetaT = thetaT.unsqueeze(0).unsqueeze(0)  # [BS 1]

		rho_ratio = ActionSignal

		if leg_flag == "F":
			boundary = self.BoundaryF(thetaT).detach().cpu().numpy()
		else:
			boundary = self.BoundaryH(thetaT).detach().cpu().numpy()
		rho_ratio = rho_ratio*0.98  # Correct TODO

		rho = rho_ratio * boundary.item()  # get rho for theta now
		return rho

	def getOvalPoint(self, theta, rho, leg_flag):
		if leg_flag == "F":
			originPoint = self.para_CF
		else:
			originPoint = self.para_CH

		trg_x = originPoint[0] + rho * math.cos(theta)
		trg_y = originPoint[1] + rho * math.sin(theta)

		return [trg_x, trg_y]

	# def getOvalPathPoint(self, rho, theta, leg_flag, halfPeriod):
	# 	'''
	# 	Input: rho from action space, theta and Central Point from Human
	# 	'''
	#
	# 	if leg_flag == "F":
	# 		if theta < halfPeriod * math.pi:
	# 			pathParameter = self.para_FU
	# 			cur_radian = theta / halfPeriod
	# 		else:
	# 			pathParameter = self.para_FD
	# 			cur_radian = (theta) / (2 - halfPeriod)
	# 	else:
	# 		if theta < halfPeriod * math.pi:
	# 			pathParameter = self.para_HU
	# 			cur_radian = theta / halfPeriod
	# 		else:
	# 			pathParameter = self.para_HD
	# 			cur_radian = (theta) / (2 - halfPeriod)
	#
	# 	originPoint = pathParameter[0]
	# 	ovalRadius = pathParameter[1]
	#
	# 	trg_x = originPoint[0] + ovalRadius[0] * math.cos(cur_radian)
	# 	trg_y = originPoint[1] + ovalRadius[1] * math.sin(cur_radian)
	#
	# 	return [trg_x, trg_y]


class LegPath2(object):
	"""docstring for ForeLegPath"""
	def __init__(self, pathType="circle", device="cpu"):
		super(LegPath2, self).__init__()
		# Trot
		# self.para_CF = [-0.00, -0.045]
		# self.para_CH = [-0.00, -0.050]
		#
		self.para_FU = [[-0.00, -0.045], [0.03, 0.01]]
		self.para_FD = [[-0.00, -0.045], [0.03, 0.005]]
		self.para_HU = [[0.00, -0.05], [0.03, 0.01]]
		self.para_HD = [[0.00, -0.05], [0.03, 0.005]]

		# self.para_FU = [[-0.00, -0.045], [0.045, 0.01]]
		# self.para_FD = [[-0.00, -0.045], [0.045, 0.015]]
		# self.para_HU = [[0.00, -0.05], [0.04, 0.01]]
		# self.para_HD = [[0.00, -0.05], [0.04, 0.02]]

	def getOvalPathPoint(self, theta, leg_flag, ActionSignal):
		'''
		Input: rho from action space, theta and Central Point from Human
		ActionSignal: 0.0-1.0
		'''
		halfPeriod = 1
		if leg_flag == "F":
			if theta < halfPeriod * math.pi:
				pathParameter = self.para_FU
				cur_radian = theta / halfPeriod
			else:
				pathParameter = self.para_FD
				cur_radian = (theta) / (2 - halfPeriod)
		else:
			if theta < halfPeriod * math.pi:
				pathParameter = self.para_HU
				cur_radian = theta / halfPeriod
			else:
				pathParameter = self.para_HD
				cur_radian = (theta) / (2 - halfPeriod)

		originPoint = pathParameter[0]
		ovalRadius = pathParameter[1]


		trg_x = originPoint[0] + ovalRadius[0] * math.cos(cur_radian) * ActionSignal
		trg_y = originPoint[1] + ovalRadius[1] * math.sin(cur_radian) * ActionSignal

		return [trg_x, trg_y]