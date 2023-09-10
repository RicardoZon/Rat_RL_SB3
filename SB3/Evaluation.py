from RatEnv.RL_wrapper3_Connect import RatRL
import gym
from stable_baselines3 import PPO
# from stable_baselines3 import SAC
# from stable_baselines3 import A2C
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from Tools.DataRecorder import DATA_Recorder

RENDER = True

if __name__ == '__main__':
    # SceneFile = "../models/dynamic_4l_t3.xml"
    # MODELPATH = "Local_Data/S0_PPO_100"
    # MODELPATH = "Local_Logs/S0_PPO_101/NAME_600000_steps"


    SceneFile = "../models/Scenario1_Planks.xml"
    # MODELPATH = "Local_Logs/S1_PPO_101/NAME_800000_steps"
    MODELPATH = "Local_Data/Wrapper3Div/S1_PPO_121"

    # SceneFile = "../models/Scenario2_Uphill.xml"  # S2
    # MODELPATH = "Local_Data/S2_PPO_102"
    # # Failled up

    # SceneFile = "../models/scene_test2pro.xml"  # S2

    # SceneFile = "../models/Scenario3_Logs.xml"  # 3
    # MODELPATH = "Local_Data/S3_PPO_103"

    # SceneFile = "../models/Scenario4_Stairs.xml"
    # MODELPATH = "Local_Data/S4_PPO_104"

    Recorder = DATA_Recorder()

    env = RatRL(SceneFile, Render=RENDER)
    # env.theController.spine_A = 30 * np.pi / 180  # SPine
    model = PPO.load(MODELPATH, env=env)

    # env = gym.make("Ant-v2")
    # model = PPO.load("data/PPO_Ant_006", env=env)
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=4)

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    pos_Ori = vec_env.envs[0].pos[1]
    pos_end = []
    for i in range(int(6000/5)):
        pos_pre = vec_env.envs[0].pos[1]

        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        print(action)
        # print(vec_env.envs[0].pos)
        # vec_env.render()
        Recorder.update(vec_env.envs[0])

        if dones[0]:
            pos_end.append(pos_pre)
            print(pos_pre)

    times = np.array(vec_env.envs[0].episode_lengths)* vec_env.envs[0].dt
    v_global = -(np.array(pos_end) - pos_Ori) / np.array(times)
    print(v_global.mean())

    # Recorder.savePath_Basic("S1_Pass_073")

