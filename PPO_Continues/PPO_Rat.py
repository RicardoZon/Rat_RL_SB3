import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous

# from RatEnv.RL_wrapper3_Connect import RatRL
from RatEnv.Wrapper_Dumpped.RL_wrapper2 import RatRL
print("Warpper 2 Native Action Framework")


RENDER_TRAIN = False
RENDER_EVAL = False

def evaluate_policy(args, env, agent, state_norm, Render=False):
    times = 1
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            # print(action)
            s_, r, done, _ = env.step(action, Render=Render)
            # env.render()  # Render
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def parserdefault():
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(2e7), help=" Maximum number of training steps")  # 3e6  DIV8
    parser.add_argument("--evaluate_freq", type=float, default=8192,
                        help="Evaluate the policy every 'evaluate_freq' steps")  # 2048 FOR DIV8
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size")  # 2048  # 1024 FOR DIV8
    parser.add_argument("--mini_batch_size", type=int, default=256, help="Minibatch size")  # 64 DIV8
    parser.add_argument("--hidden_width", type=int, default=256,  # 64 DIV8
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
    return args

if __name__ == '__main__':
    args = parserdefault()
    # main(args, number=3)
    RESUME = True
    ACTORPATH = "./data_train/PPO_Rat_env_PlaneNPPO_number_108.pth"

    number = 112
    print("Number="+str(number))

    SceneFile = "../models/dynamic_4l_t3.xml"
    SceneName = "PlaneNPPO"  # Plane S0

    # SceneFile = "../models/Scenario1_Planks.xml"  # Now: 2
    # SceneName = "S1NPPO"  # SceneName = "S1"
    #
    # SceneFile = "../models/Scenario2_Uphill.xml"  #Now: 2
    # SceneName = "S2"

    # SceneFile = "../models/Scenario3_Logs.xml"  # Now: 3
    # SceneName = "S3"

    # SceneFile = "../models/Scenario4_Stairs.xml"
    # SceneName = "stairNPPO"

    ##################################################################################################
    env = RatRL(SceneFile, Render=RENDER_TRAIN)
    env_evaluate = RatRL(SceneFile, Render=RENDER_EVAL)  # When evaluating the policy, we need to rebuild an environment

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(SceneName))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    # print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    max_eval_R = -500.0
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_Rat/env_{}_{}_number_{}'.format(SceneName, args.policy_dist, number))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    # Load new
    if RESUME:
        checkpoint = torch.load(ACTORPATH)
        agent.actor.load_state_dict(checkpoint['state_dict_actor'])
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        agent.critic.load_state_dict(checkpoint['state_dict_critic'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        state_norm.running_ms = checkpoint['state_norm_running']

    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        print("Reset on {}".format(total_steps))
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action, Render=RENDER_TRAIN)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # print(total_steps)

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm, Render=RENDER_EVAL)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(SceneName), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    # np.save('./data_train/PPO_Rat_env_{}_number_{}.npy'.format(SceneName, number), np.array(evaluate_rewards))
                    state = {
                        'total_steps': total_steps,
                        'state_dict_actor': agent.actor.state_dict(),
                        'optimizer_actor': agent.optimizer_actor.state_dict(),
                        'state_dict_critic': agent.critic.state_dict(),
                        'optimizer_critic': agent.optimizer_critic.state_dict(),
                        # "state_norm_mean": state_norm,
                        # "state_norm_std": state_norm.running_ms.std,
                        "state_norm_running":state_norm.running_ms
                    }
                    torch.save(state, './data_train/PPO_Rat_env_{}_number_{}.pth'.format(SceneName, number))
                # if evaluate_reward > max_eval_R:
                #     max_eval_R = evaluate_reward
                #     state = {
                #         'total_steps': total_steps,
                #         'state_dict_actor': agent.actor.state_dict(),
                #         'optimizer_actor': agent.optimizer_actor.state_dict(),
                #         'state_dict_critic': agent.critic.state_dict(),
                #         'optimizer_critic': agent.optimizer_critic.state_dict(),
                #         "state_norm_running": state_norm.running_ms
                #     }
                #     torch.save(state, './data_train/PPO_Rat_env_{}_number_{}_BEST.pth'.format(SceneName, number))



    print("Number={}".format(number))
