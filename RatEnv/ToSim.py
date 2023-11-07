from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import deque
import scipy.io as scio
import time
from RatEnv.Controller import MouseController
from RatEnv.Controller import MouseControllerB
import argparse
from mujoco_py.generated import const

# 理念：
# SimModel是对Mouse的虚拟模拟

# DEBUG = True

class SimModel(object):
	"""docstring for SimModel"""
	def __init__(self, modelPath, Render=False):
		super(SimModel, self).__init__()
		self.model = load_model_from_path(modelPath)
		self.sim = MjSim(self.model)

		if Render:
			# render must be called mannually
			self.viewer = MjViewer(self.sim)
			self.viewer.cam.azimuth = 0
			self.viewer.cam.lookat[0] += 0.25
			self.viewer.cam.lookat[1] += -0.5
			self.viewer.cam.distance = self.model.stat.extent * 0.5

		self.sim_state = self.sim.get_state()
		self.sim.set_state(self.sim_state)
		self.legPosName = [
			["router_shoulder_fl", "foot_s_fl"],
			["router_shoulder_fr", "foot_s_fr"],
			["router_hip_rl", "foot_s_rl"],
			["router_hip_rr", "foot_s_rr"]]
		self.fixPoint = "body_ss" #"neck_ss"

		self.legRealPoint_x = [[],[],[],[]]
		self.legRealPoint_y = [[],[],[],[]]
		self.movePath = [[],[],[]]
		self.legRealPoint_x_ori = [[], [], [], []]
		self.legRealPoint_y_ori = [[], [], [], []]

		self.imu_pos = deque([])
		self.imu_quat = deque([])
		self.imu_vel = deque([])
		self.imu_acc = deque([])
		self.imu_gyro = deque([])

		self.rendered = False

	def initializing(self):
		self.sim.set_state(self.sim_state)
		self.movePath = [[],[],[]]
		self.legRealPoint_x = [[],[],[],[]]
		self.legRealPoint_y = [[],[],[],[]]
		self.legRealPoint_x_ori = [[], [], [], []]
		self.legRealPoint_y_ori = [[], [], [], []]

		self.imu_pos = deque([])
		self.imu_quat = deque([])
		self.imu_vel = deque([])
		self.imu_acc = deque([])
		self.imu_gyro = deque([])

	def runStep(self, ctrlData, render=False, legposcal=False):
		# ------------------------------------------ #
		# ID 0, 1 left-fore leg and coil
		# ID 2, 3 right-fore leg and coil
		# ID 4, 5 left-hide leg and coil
		# ID 6, 7 right-hide leg and coil
		# Note: For leg, it has [-1: front; 1: back]
		# Note: For fore coil, it has [-1: leg up; 1: leg down]
		# Note: For hide coil, it has [-1: leg down; 1: leg up]
		# ------------------------------------------ #
		# ID 08 is neck		(Horizontal)
		# ID 09 is head		(vertical)
		# ID 10 is spine	(Horizontal)  [-1: right, 1: left]
		# Note: range is [-1, 1]
		# ------------------------------------------ #
		self.sim.data.ctrl[:] = ctrlData
		self.sim.step()
		if render:
			self.viewer.render()

		# 记录数据
		if legposcal:
			tData = self.sim.data.get_site_xpos(self.fixPoint)
			for i in range(3):
				self.movePath[i].append(tData[i])
			for i in range(4):
				originPoint = self.sim.data.get_site_xpos(self.legPosName[i][0])
				currentPoint = self.sim.data.get_site_xpos(self.legPosName[i][1])
				#print(originPoint, currentPoint)
				tX = currentPoint[1]-originPoint[1]
				tY = currentPoint[2]-originPoint[2]
				self.legRealPoint_x[i].append(tX)
				self.legRealPoint_y[i].append(tY)
				self.legRealPoint_x_ori[i].append(originPoint[1])
				self.legRealPoint_y_ori[i].append(originPoint[2])

		# imudata
		pos = self.sim.data.sensordata[16:16+3]  # com_pos from imu    imu_pos
		self.imu_pos.append(list(pos))

		self.pos = list(pos)
		self.quat = list(self.sim.data.sensordata[19:19+4])
		self.vel = list(self.sim.data.sensordata[23:23+3])
		self.acc = list(self.sim.data.sensordata[26:26+3])
		self.gyro = list(self.sim.data.sensordata[29:29 + 3])

		# DEBUG = False
		# if DEBUG:
		# 	print(self.imu_pos.__len__())


	def getTime(self):
		return self.sim.data.time

	def point_distance_line(self, point,line_point1,line_point2):
		vec1 = line_point1 - point
		vec2 = line_point2 - point
		distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
		return distance

	def drawPath(self):
		path_X = self.movePath[0]
		path_Y = self.movePath[1]
		tL = len(path_X)

		ds = 1
		dL = int(tL/ds)
		check_x = []
		check_y = []
		print(tL)
		for i in range(dL):
			check_x.append(path_X[i*ds])
			check_y.append(path_Y[i*ds])

		check_x.append(path_X[-1])
		check_y.append(path_Y[-1])

		dX = path_X[0]-path_X[-1]
		dY = path_Y[0]-path_Y[-1]
		dis = math.sqrt(dX*dX + dY*dY)
		print("Dis --> ", dis)

		start_p = np.array([check_x[0], check_y[0]])
		end_p = np.array([check_x[-1], check_y[-1]])

		maxDis = 0
		for i in range(tL):
			cur_p = np.array([path_X[i], path_Y[i]])
			tDis = self.point_distance_line(cur_p, start_p, end_p)
			if tDis > maxDis:
				maxDis = tDis
		print("MaxDiff --> ", maxDis)
		plt.plot(path_X, path_Y)
		plt.plot(check_x, check_y)
		plt.grid()
		plt.show()

		return dis

	def savePath(self, flag):
		filePath = "Data/path_"+flag+".txt"
		trajectoryFile = open(filePath, 'w')
		dL = len(self.movePath[0])
		for i in range(dL):
			for j in range(3):
				trajectoryFile.write(str(self.movePath[j][i])+' ')
			trajectoryFile.write('\n')
		trajectoryFile.close()


if __name__ == '__main__':
	RENDER = True
	MODELPATH = "../models/dynamic_4l_t3.xml"
	# MODELPATH = "../models/Scenario1_Planks.xml"
	# MODELPATH = "../models/Scenario4_Stairs.xml"
	# MODELPATH = "../models/Scenario4_Stairs_Sparse.xml"

	RUN_STEPS = 40000

	theMouse = SimModel(MODELPATH, Render=RENDER)
	frame_skip = 1
	dt = theMouse.model.opt.timestep*frame_skip
	fre_cyc = 0.67  # 1.25  # 0.80?
	SteNum = int(1 / (dt * fre_cyc) / 2)  # /1.25)
	theController = MouseController(SteNum=SteNum)
	# theController.pathStore.para_FU = [[0.005, -0.045], [0.03, 0.01]]
	# theController.pathStore.para_FD = [[0.005, -0.045], [0.03, 0.002]]
	# theController.pathStore.para_HU = [[0.005, -0.055], [0.03, 0.025]]
	# theController.pathStore.para_HD = [[0.005, -0.055], [0.03, 0.002]]
	# theController.spine_A = 30 * np.pi / 180
	# coms = deque(maxlen=600)  # Trace show

	for i in range(500):
		ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0,-1.2, 0,0,0,0]
		theMouse.runStep(ctrlData)
	theMouse.initializing()
	start = time.time()

	def run_tmp():
		for i in range(RUN_STEPS):
			pos_pre = theMouse.pos.copy()
			ctrlData = theController.runStep()				# No Spine
			#tCtrlData = theController.runStep_spine()		# With Spine
			for _ in range(frame_skip):
				theMouse.runStep(ctrlData, render=RENDER)
			pos = theMouse.pos

			v = (pos[1]-pos_pre[1])*(-4)/dt
			# print(v)
			# print(pos)
			# print(theMouse.sim.data.get_joint_qpos("knee1_fl"))
			# print(theMouse.sim.data.get_joint_qpos("ankle_fl"))

	run_tmp()
	end = time.time()
	timeCost = end-start
	print("Time -> ", timeCost)
	# print(pos)

	'''
	s_trap = theMouse.sim.get_state()
	torch.save(s_trap, './XXX.pth')
	'''