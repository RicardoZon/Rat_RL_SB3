from RatEnv.RL_wrapper3_Connect import RatRL
import gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np

RENDER_TRAIN = False

if __name__ == '__main__':
    # SceneFile = "../models/dynamic_4l_t3.xml"
    # NAME = "S0_PPO_Spine_200"  # Plane S0
    # NAME = "S0_PPO_120_B2"
    # NAME = "S0_PPO_120_B3"

    SceneFile = "../models/Scenario1_Planks.xml"  # S1
    NAME = "S1_PPO_Spine_201"
    # NAME = "S1_PPO_121_B2"
    # NAME = "S1_PPO_121_B3"

    # SceneFile = "../models/Scenario2_Uphill.xml"  # S2
    # NAME = "S2_PPO_Spine_202"
    # NAME = "S2_PPO_122_B2"
    # NAME = "S2_PPO_122_B3"


    # SceneFile = "../models/Scenario3_Logs.xml"  # S3
    # NAME = "S3_PPO_Spine_203"
    # NAME = "S3_PPO_123_B2"
    # NAME = "S3_PPO_123_B3"

    # SceneFile = "../models/Scenario4_Stairs.xml"  # S4
    # NAME = "S4_PPO_Spine_204"
    # NAME = "S4_PPO_124_B2"
    # NAME = "S4_PPO_124_B3"

    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path="./Local_Logs/Wrapper3Div/" + NAME,
        name_prefix="NAME",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    env = RatRL(SceneFile, Render=RENDER_TRAIN)
    env.theController.spine_A = 30 * np.pi / 180  # SPine
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./Local_Rat_Tensorboard/Wrapper3Div/")
    # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    model.learn(total_timesteps=2_000_000, tb_log_name=NAME, reset_num_timesteps=True,
                callback=checkpoint_callback)
    model.save("./Local_Data/Wrapper3Div/" + NAME)