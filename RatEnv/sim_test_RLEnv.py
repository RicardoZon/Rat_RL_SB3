from RL_wrapper2_Dir import RatRL
RUN_STEPS = 4000
# RUN_STEPS = 50  # Half Per Action
# SceneName = "../models/dynamic_4l_t3.xml"
SceneName = "../models/dynamic_4l_t3_Change.xml"
RENDER = True

if __name__ == '__main__':
    env = RatRL(SceneName, Render=RENDER, timestep=0.01)
    s = env.reset()
    for _ in range(5000):
        env.step([0., 0., 0., 0., 0., 0., 0., 0.])