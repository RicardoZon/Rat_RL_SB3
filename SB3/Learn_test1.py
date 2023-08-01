from RatEnv.RL_wrapper3_Connect import RatRL
import gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import A2C

RENDER_TRAIN = False

if __name__ == '__main__':
    # SceneFile = "../models/dynamic_4l_t3.xml"
    # SceneFile = "../models/scene_test2.xml"  # S2
    # SceneFile = "../models/scene_test3.xml"  # S3
    # SceneFile = "../models/S4_stair.xml"  # S4
    # SceneName = "PlanePPO"  # Plane S0


    # SceneFile = "../models/scene_test1.xml"  # S1
    # # NAME = "S1_PPO_010"
    # # NAME = "S1_SAC_011"
    # NAME = "S1_A2C_012"
    #
    # env = RatRL(SceneFile, Render=RENDER_TRAIN)
    # # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # model.learn(total_timesteps=1_000_000, tb_log_name=NAME)
    # model.save("./data/" + NAME)


    # SceneFile = "../models/scene_test2.xml"  # S2
    # # NAME = "S2_PPO_013"
    # # NAME = "S2_SAC_014"
    # NAME = "S2_A2C_015"
    #
    # env = RatRL(SceneFile, Render=RENDER_TRAIN)
    # # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # model.learn(total_timesteps=1_000_000, tb_log_name=NAME)
    # model.save("./data/" + NAME)


    # SceneFile = "../models/scene_test3.xml"  # S3
    # # NAME = "S3_PPO_016"
    # # NAME = "S3_SAC_017"
    # NAME = "S3_A2C_018"
    #
    # env = RatRL(SceneFile, Render=RENDER_TRAIN)
    # # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # model.learn(total_timesteps=1_000_000, tb_log_name=NAME)
    # model.save("./data/" + NAME)


    SceneFile = "../models/scene_S4_stair.xml"  # S4
    # NAME = "S4_PPO_019"
    # NAME = "S4_SAC_020"
    NAME = "S4_A2C_021"

    env = RatRL(SceneFile, Render=RENDER_TRAIN)
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    model.learn(total_timesteps=1_000_000, tb_log_name=NAME)
    model.save("./data/" + NAME)





    # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # model.learn(total_timesteps=1_000_000, tb_log_name="SAC_Plane_002_Try")
    # model.save("./data/SAC_Plane_002_Try")

    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./Rat_Tensorboard/")
    # model.learn(total_timesteps=1_000_000, tb_log_name="A2C_Plane_005_Try")
    # model.save("./data/A2C_Plane_005_Try")

    # del model
