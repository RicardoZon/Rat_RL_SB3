import argparse

from ToSim import SimModel
from Controller import MouseController
import matplotlib.pyplot as plt
import time


RUN_STEPS = 10000
if __name__ == '__main__':
	parser = argparse.ArgumentParser("Description.")
	parser.add_argument('--fre', default=0.67,
		type=float, help="Gait stride")
	args = parser.parse_args()

	theMouse = SimModel("../models/dynamic_4l_t3.xml", Render=True)
	frame_skip = 1
	dt = theMouse.model.opt.timestep*frame_skip

	theController = MouseController(args.fre, timestep=dt)

	for i in range(500):
		ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0,-1.2, 0,0,0,0]
		theMouse.runStep(ctrlData)
	theMouse.initializing()
	start = time.time()

	for i in range(RUN_STEPS):
		pos_pre = theMouse.pos.copy()
		ctrlData = theController.runStep()				# No Spine
		#tCtrlData = theController.runStep_spine()		# With Spine
		for _ in range(frame_skip):
			theMouse.runStep(ctrlData, render=True)
		pos = theMouse.pos

		v = (pos[1]-pos_pre[1])*(-4)/dt
		# print(v)
		# print(pos)
		print(theMouse.sim.data.get_joint_qpos("knee1_fl"))
		print(theMouse.sim.data.get_joint_qpos("ankle_fl"))

	end = time.time()
	timeCost = end-start
	print("Time -> ", timeCost)