import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        self.fc4 = nn.Linear(state_size, fc1_units)
        self.fc5 = nn.Linear(fc1_units, fc2_units)
        self.fc6 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x1 = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)
        
        x2 = F.relu(self.fc4(state))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)
        return x1, x2

    def Q1(self, state):
        x1 = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)
        return x1



class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, clip, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.softmax_layer = nn.Softmax( dim=1) 
        self.clip = clip
    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        mu = torch.argmax(logits, dim=1)
        action_prob = self.softmax_layer(logits) 
        action_prob = action_prob + torch.finfo(torch.float32).eps
        log_action_prob = torch.log(action_prob)
        log_action_prob = torch.clamp(log_action_prob, min= self.clip, max=0)
        policy_dist = Categorical(logits=logits)
        pi = policy_dist.sample() 
        return mu, pi, action_prob, log_action_prob, logits
