from RatEnv.RL_wrapper3_Connect import RatRL
import gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback

RENDER_TRAIN = False

if __name__ == '__main__':
    # SceneFile = "../models/dynamic_4l_t3.xml"
    # NAME = "S0_PPO_101"  # Plane S0

    # SceneFile = "../models/scene_test1.xml"  # S1
    # NAME = "S1_PPO_102"

    # SceneFile = "../models/scene_test2.xml"  # S2
    # NAME = "S2_PPO_103"

    # SceneFile = "../models/scene_test3.xml"  # S3
    # NAME = "S3_PPO_104"

    SceneFile = "../models/scene_S4_stair.xml"  # S4
    NAME = "S4_PPO_105"

    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path="./Local_Logs/" + NAME,
        name_prefix="NAME",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    env = RatRL(SceneFile, Render=RENDER_TRAIN)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./Local_Rat_Tensorboard/")
    # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    model.learn(total_timesteps=2_000_000, tb_log_name=NAME, reset_num_timesteps=True,
                callback=checkpoint_callback)
    model.save("./Local_Data/" + NAME)