from RL_wrapper3_Connect import RatRL
import matplotlib.pyplot as plt
from Tools.PlotTool import plot_simple

RUN_STEPS = 4000
# RUN_STEPS = 50  # Half Per Action
# SceneName = "../models/dynamic_4l_t3.xml"
SceneName = "../models/dynamic_4l_t3_Change.xml"
# SceneName = "../models/scene_test3.xml"


# SceneName = "../models/scene_test1.xml"


if __name__ == '__main__':
    RENDER = False

    env = RatRL(SceneName, Render=RENDER)
    R = []
    V_vels = []
    V_x = []
    V_z = []

    Rats = []

    s = env.reset()
    for _ in range(RUN_STEPS):
        # theta
        action = [1, 1, 1, 1]  # FL, FR, HL, HR  [-1, 1]-->[0,1]:  (a+1)/2
        observation, reward, done, info = env.step(action, Render=RENDER)
        # env.render()

        V_vels.append(-env.Vels_mean[1]*4)
        V_x.append(env.Vels_mean[0])
        V_z.append(env.Vels_mean[2])
        # Delta_vels.append(env.Delta_vel)
        R.append(reward)
        Rats.append(env.rat)
        print(env.theController.curStep)
        # plot_simple([info])


    # plot_simple([V_vels[100:3000], Delta_vels[100:3000],
    #              R[100 + env.Window_Cover:3000 + +env.Window_Cover]])

    # plot_simple([V_vels[:], R[:], Rats],
    #             leg=['V_vels', 'R', 'rats'])

    # plot_simple([V_x, V_z],
    #             leg=['V_x', 'V_z'])


