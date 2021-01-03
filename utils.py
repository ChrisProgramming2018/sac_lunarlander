import os
import gym
import random
import torch
import numpy as np
from collections import deque
from torch import distributions
import torch.nn.functional as F


class GumbelSoftmax(distributions.RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size()):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        '''
        Gumbel-softmax resampling using the Straight-Through trick.
        Credit to Ian Temple for bringing this to our attention. To see standalone code of how this works, refer to https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
        '''
        rout = super().rsample(sample_shape)  # differentiable
        out = F.one_hot(torch.argmax(rout, dim=-1), self.logits.shape[-1]).float()
        return (out - rout).detach() + rout

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)



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
            action = agent.act(state, 0)
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
