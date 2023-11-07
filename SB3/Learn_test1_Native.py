# from RatEnv.RL_wrapper2_MujoEnv_Compare import RatRL
from RatEnv.RL_wrapper3_Connect_Compare import RatRL
# from RatEnv.RL_wrapper3_Connect_Compare_SensorDir import RatRL
import gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import A2C
import warnings
from stable_baselines3.common.callbacks import CheckpointCallback

RENDER_TRAIN = False

if __name__ == '__main__':
    # SceneFile = "../models/dynamic_4l_t3.xml"
    # NAME = "S0_PPO_NativeWrap3_158Vbias"
    # NAME = "S0_SAC_NativeWrap3_159"
    # NAME = "S0_A2C_NativeWrap3_160"
    # NAME = "S0_PPO_NativeWrap3_180"

    SceneFile = "../models/Scenario1_Planks.xml"  # S1
    NAME = "S1_PPO_NativeWrap3_167_B3"  # B3TODO
    # NAME = "S1_SAC_NativeWrap3_168_B3"
    # NAME = "S1_A2C_NativeWrap3_169_B3"

    # SceneFile = "../models/Scenario2_Uphill.xml"  # S2  Spe
    # NAME = "S2_PPO_NativeWrap3_170_B3"
    # NAME = "S2_SAC_NativeWrap3_171_B3"
    # NAME = "S2_A2C_NativeWrap3_172_B3"

    # SceneFile = "../models/Scenario3_Logs.xml"  # S3
    # NAME = "S3_PPO_NativeWrap3_173_B3"
    # NAME = "S3_SAC_NativeWrap3_174_B3"
    # NAME = "S3_A2C_NativeWrap3_175_B3"

    # SceneFile = "../models/Scenario4_Stairs.xml"  # S4
    # NAME = "S4_PPO_NativeWrap3_176_B3"
    # NAME = "S4_SAC_NativeWrap3_177_B3"
    # NAME = "S4_A2C_NativeWrap3_178_B3"

    # warnings.filterwarnings("ignore")
    # checkpoint_callback = CheckpointCallback(
    #     save_freq=200_000,
    #     save_path="./Local_Logs/Wrapper3Div/" + NAME,
    #     name_prefix="NAME",
    #     save_replay_buffer=True,
    #     save_vecnormalize=False,
    # )

    env = RatRL(SceneFile, Render=RENDER_TRAIN)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./Local_Rat_Tensorboard/Wrapper3Div/")
    # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./Local_Rat_Tensorboard/Wrapper3Div/")
    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./Local_Rat_Tensorboard/Wrapper3Div/")
    model.learn(total_timesteps=2_000_000, tb_log_name=NAME, reset_num_timesteps=True)
    model.save("./Local_Data/Wrapper3Div/" + NAME)

    # del model
