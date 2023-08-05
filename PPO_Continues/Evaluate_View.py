import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
from collections import deque
from Tools.PlotTool import plot_simple
import scipy.io as scio
from RatEnv.RL_wrapper3_Connect_Resume import RatRL
# from RatEnv.RL_wrapper3_Connect import RatRL
from PPO_Rat import evaluate_policy
from Tools.DataRecorder import DATA_Recorder
import time

# SceneFile = "../models/dynamic_4l_t3.xml"
# ACTORPATH = "./Local_Data/PPO_Rat_env_Plane_number_57.pth"
# SceneName = "Plane"

SceneFile = "../models/Scenario1_Planks.xml"
ACTORPATH = "./Local_Data/PPO_Rat_env_S1_number_60_BEST.pth"
SceneName = "S1"

# SceneFile = "../models/Scenario2_Uphill.xml"
# ACTORPATH = "./data_train/PPO_Rat_env_S2_number_62_BEST.pth"

# SceneFile = "../models/scene_test2pro.xml"
# ACTORPATH = "./data_train/PPO_Rat_env_S2P_number_64.pth"

# SceneFile = "../models/Scenario3_Logs.xml"
# ACTORPATH = "./data_train/PPO_Rat_env_S3_number_67.pth"
# SceneName = "S3"

# SceneFile = "../models/dynamic_4l_t3.xml"
# ACTORPATH = "./data_train/PPO_Rat_env_PlaneNPPO_number_108.pth"
# SceneName = "NPPOPLANE"

RENDER_EVAL = True


if __name__ == '__main__':
    # SceneName = "S1"
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1024,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")  # 2048
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")

    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    # env = RatRL(SceneFile)
    env_evaluate = RatRL(SceneFile, Render=RENDER_EVAL)  # When evaluating the policy, we need to rebuild an environment

    args.state_dim = env_evaluate.observation_space.shape[0]
    args.action_dim = env_evaluate.action_space.shape[0]
    args.max_action = float(env_evaluate.action_space.high[0])
    args.max_episode_steps = env_evaluate._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(SceneName))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))

    evaluate_num = 0  # Record the number of evaluations
    # evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization

    # Load Old
    # agent.actor.load_state_dict(torch.load(ACTORPATH))

    # Load new
    checkpoint = torch.load(ACTORPATH)
    print("Step={}".format(checkpoint['total_steps']))
    agent.actor.load_state_dict(checkpoint['state_dict_actor'])
    state_norm.running_ms = checkpoint['state_norm_running']
    actions = []
    actions.append(args.action_dim*[1.0])  # Init Action
    rewards = []


    # agent.actor.eval()  # 不启用 BatchNormalization 和 Dropout
    # evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm, Render=RENDER_EVAL)

    gyros_mean = deque([])
    vels_mean = deque([])

    s = env_evaluate.reset()
    if args.use_state_norm:
        s = state_norm(s, update=False)  # During the evaluating,update=False
    done = False
    episode_reward = 0
    time_start = time.time()
    while not done:
        a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
        if args.policy_dist == "Beta":
            action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
        else:
            action = a
        # action = [1., 1., 1., 1.]
        s_, r, done, _ = env_evaluate.step(action, LegCal=True)
        # s_, r, done, _ = env_evaluate.step(action, LegCal=True, Render=True)
        # env.render()  # Render
        if args.use_state_norm:
            s_ = state_norm(s_, update=False)
        episode_reward += r
        s = s_

        gyros_mean.append(env_evaluate.Gyros_mean)
        vels_mean.append(env_evaluate.Vels_mean)

        print([action, r])
        actions.append(action)
        rewards.append(r)
    print(time.time()-time_start)


    print(episode_reward)
    # print("Time:{}--Pos:{}".format(env_evaluate.theMouse.getTime(), env_evaluate.theMouse.pos))
    gyros_mean = np.array(gyros_mean)

    # plot_simple([])

    # New1
    import matplotlib.pyplot as plt

    Acts = np.array(actions).transpose()
    # times = np.arange(0, 200 * 0.002 * 46, 0.002 * 46)[0:200]
    times = np.arange(0, 200 * 0.002 * 46, 0.002 * 46)


    plt.figure()
    plt.plot(times, Acts[0])
    plt.plot(times, Acts[1])
    plt.plot(times, Acts[2])
    plt.plot(times, Acts[3])
    plt.legend(["FL", "FR", "HL", "HR"])
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(times[0:200], rewards)
    plt.legend(["R"])
    plt.grid()
    plt.show()

    # FileName = "actions_" + SceneName
    # scio.savemat(FileName + '.mat', {'actions': Acts})  # 写入mat文件

    # RECORD = DATA_Recorder()
    # RECORD.savePath_TOSIM("trag_S1_E60Best", env_evaluate.theMouse)





