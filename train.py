from stable_baselines import HER, SAC, DQN, DDPG, TD3
from stable_baselines.common.atari_wrappers import FrameStack
import three_finger.envs

import click
import gym
import os
import os.path as osp
import time

ALGS = {'sac': SAC, 'ddpg': DDPG, 'td3': TD3}

@click.command()
@click.option('--env_name', type=str, default='Gripper2DSamplePoseEasy-v2',
        help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--exp_name', type=str, default=None, help='exp directory name') 
@click.option('--n_steps', type=float, default=1e6, help='the number of training steps to run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--lr', type=float, default=1e-3, help='learning rate')
@click.option('--alg', type=str, default='sac', help='alg to use (sac, ddpg, td3)')
@click.option('--gamma', type=float, default=0.99, help='discount factor')
@click.option('-l', '--num_layers', type=int, default=3, help='num hidden layers')
@click.option('-hu', '--layer_size', type=int, default=256, help='hidden layer size')

@click.option('--drop_pen', type=int, default=-500, help='drop penalty') 

def train(env_name, exp_name, n_steps, seed, lr, alg, gamma, num_layers,
          layer_size, drop_pen):
    exp_root = './experiments'
    hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    if exp_name is None:
        exp_name = '_'.join(['HER', alg, env_name.split('-')[0], str(n_steps)])

    exp_dir = osp.join(exp_root, exp_name, hms_time)
    os.makedirs(exp_dir)

    env = gym.make(env_name, reward_type='sparse', drop_penalty=-abs(drop_pen))

    model = HER('MlpPolicy', env, ALGS.get(alg.lower()), n_sampled_goal=4,
                tensorboard_log=exp_dir,
                seed=seed,
                goal_selection_strategy='future',
                verbose=1, buffer_size=int(1e6),
                learning_rate=lr,
                gamma=gamma, batch_size=256,
                policy_kwargs=dict(layers=[layer_size]*num_layers))

    # Train for n_steps
    model.learn(int(n_steps),)
    # Save the trained agent
    model.save(exp_dir)

if __name__ == '__main__':
    train()
