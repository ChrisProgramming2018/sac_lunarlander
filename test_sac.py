import atari_py
import gym
from model import Actor
import sys
import torch
import numpy as np
from torch.nn import Module, Linear
from torch.distributions import Distribution, Normal, Categorical
from torch.nn.functional import relu, logsigmoid
env = gym.make("LunarLander-v2")

LOG_STD_MIN_MAX = (-20, 2)
state = env.reset()
score = 0
t = 0
DEVICE = "cpu"




state_size = env.observation_space.shape[0]
action_size = env.action_space.n
seed = 2
actor = Actor(state_size, action_size, seed)

state = torch.Tensor(state).unsqueeze(0)
print(state.shape)
action_dim = 1
probs =  actor(state)

print("probs ", probs)
m = Categorical(probs)
action = m.sample()
print("m ", m)
print("action", action)
sys.exit()

while True:
    t += 1
    action = env.action_space.sample()
    next_state, reward, done,_ = env.step(action)
    score += reward
    env.render()
    if done:
        print("Episode reward {} after {} steps ".format(score, t))
        env.close()
        break
