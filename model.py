import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class RQNetwork(nn.Module):
    """Critic Q-funct Model."""

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
        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        self.fc4 = nn.Linear(state_size + action_size, fc1_units)
        self.fc5 = nn.Linear(fc1_units, fc2_units)
        self.fc6 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """Build a network that maps state -> action values."""
        xu = torch.cat([state, action], dim=1)
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)
        
        x2 = F.relu(self.fc4(xu))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)
        return x1, x2

    def Q1(self, state, action):
        xu = torch.cat([state, action], dim=1)
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)
        return x1


class QNetwork(nn.Module):
    """Critic Q-func Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, fc3_units=32):
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
        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        
        self.fc5 = nn.Linear(state_size + action_size, fc1_units)
        self.fc6 = nn.Linear(fc1_units, fc2_units)
        self.fc7 = nn.Linear(fc2_units, fc3_units)
        self.fc8 = nn.Linear(fc3_units, 1)

    def forward(self, state, action):
        """Build a network that maps state -> action values."""
        xu = torch.cat([state, action], dim=1)
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.fc4(x1)
        
        x2 = F.relu(self.fc5(xu))
        x2 = F.relu(self.fc6(x2))
        x2 = F.relu(self.fc7(x2))
        x2 = self.fc8(x2)
        return x1, x2

    def Q1(self, state, action):
        xu = torch.cat([state, action], dim=1)
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.fc4(x1)
        return x1

class RActor(nn.Module):
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
    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, clip, fc1_units=64, fc2_units=64, fc3_units=32):
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
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.fc4(x)
        return logits
