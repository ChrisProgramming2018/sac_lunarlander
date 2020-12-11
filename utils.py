import os
import gym
import random
import torch
import numpy as np
from collections import deque


def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname 
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

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



def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')


def eval_policy(env, agent, writer, steps, config, episodes=3):
    average_score = 0
    average_steps = 0
    print("Eval Policy")
    for i in range(episodes):
        env = gym.wrappers.Monitor(env,str(config["locexp"])+"/vid/{}/{}".format(steps, i), video_callable=lambda episode_id: True,force=True)
        env.seed(i)
        score = 0 
        state = env.reset()
        t = 0
        while True:
            t += 1
            action = agent.act_eps(state)
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                average_score += score
                average_steps += t
                break 
    average_score /= episodes
    average_steps /= episodes
    print("Evaluate policy on {} Episodes".format(episodes))   
    print("Average steps  {} ".format(average_steps))   
    print("Average score  {:.2f} ".format(average_score))   
