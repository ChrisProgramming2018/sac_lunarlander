import gym
import random
import torch
import numpy as np
from sac_agent import SACAgent
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import time
from datetime import datetime
from collections import namedtuple, deque
from utils import eval_policy, mkdir, write_into_file



def time_format(sec):
    """

    Args:
        param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)


def main(args):
    with open (args.param, "r") as f:
        config = json.load(f)
    env = gym.make('LunarLander-v2')
    #env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
    env.seed(config['seed'])
    config["target_entropy"] =  np.log(env.action_space.n)
    config["runs"] =  args.runs
    print("target entropy ",config["target_entropy"])
    if args.mode == "debug":
        config["batch_size"] = args.batch_size

    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    agent = SACAgent(state_size=8, action_size=4, config=config)
    replay_buffer = ReplayBuffer((8, ), (1, ), int(args.buffer_size),  config['device'])
    # agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    n_episodes = 2000
    scores = []
    # list containing scores from each episode
    scores_window = deque(maxlen=100)
    # eps = 1   # random policy
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    pathname = dt_string + "seed_" + str(config['seed'])
    tensorboard_name = "result_"+ str(config["runs"]) +  'runs/' + pathname
    writer = SummaryWriter(tensorboard_name)
    best_reward = - 300
    best_step = 0
    #eval_policy(env, agent, writer, 0, config)
    t0 = time.time()
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        env_score = 0
        for t in range(args.max_episode_steps):    
            _, action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(replay_buffer, writer)
            env_score += reward
            done_bool = 0 if t + 1 == args.max_episode_steps else float(done)
            replay_buffer.add(state, action, reward, next_state, done, done_bool)
            state = next_state
            if done:
                break
        if env_score > best_reward:
            best_reward = env_score
            best_step = i_episode
        scores_window.append(env_score)
        mean_reward =  np.mean(scores_window)
        writer.add_scalar('env_reward', mean_reward, i_episode)
        print('\rEpisode {}\tAverage Score: {:.2f} Time: {}'.format(i_episode, np.mean(scores_window),  time_format(time.time()-t0)))
        if i_episode % 1000 == 0:
            eval_policy(env, agent, writer, i_episode, config)
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes! \tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    
    print("done training write in file")
    text = "Run {}  best r {} at {} ".format(config['run'], best_reward , best_step)
    filepath = "results"
    mkdir("", filepath)
    write_into_file(filepath, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--fc1_units', default=256, type=int)
    parser.add_argument('--fc2_units', default=256, type=int)
    parser.add_argument('--mode', default="dqn", type=str)
    parser.add_argument('--buffer_size', default=1e5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_episode_steps', default=1000, type=int) 
    parser.add_argument('--runs', default=1, type=int) 
    arg = parser.parse_args()
    main(arg)
