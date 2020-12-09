import numpy as np
import sys
import random
from collections import namedtuple, deque

from model import QNetwork, Actor

import torch
import torch.nn.functional as F
import torch.optim as optim



class SACAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = int(config["seed"])
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.device = config['device']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.ddqn = config['DDQN']
        print("seed", self.seed)
        # Q-Network
        self.target_entropy = config["target_entropy"]
        self.target_entropy = 0.416
        
        self.actor = Actor(state_size, action_size, self.seed, config["clip"]).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.qnetwork_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=config["device"])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config["lr_alpha"])
        self.t_step = 0
        self.update_freq = config["update_freq"]
    
    def step(self, memory, writer):
        self.t_step += 1 
        if self.t_step % self.update_freq == 0:
            if len(memory) > self.batch_size:
                states, actions, rewards, next_states, dones = memory.sample(self.batch_size)
                self.update_q(states, actions, rewards, next_states, dones, writer)

    def act(self, state):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            mu, pi, action_prob, log_action_prob, logits = self.actor(state)
        self.actor.train()
        return mu.item(), pi.item()
    
    def update_q(self, states, actions, rewards, next_states, dones, writer):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        alpha = torch.exp(self.log_alpha) * 0
        # update the q function by mean squard error of 
        # q local s_t, a - target 
        # target r + yEV(s_t+1)  v(s_t) = pi(st)  Q(s_t,a_t) - alpha log(pi(a|s)
        with torch.no_grad():
            #print(next_states.shape) 
            mu, next_actions, action_prob_next, next_log_probs, logits = self.actor(next_states)
            #print(mu)
            #print(mu.shape)
            #sys.exit() 
            #next_log_probs = next_log_probs.gather(1, next_actions.unsqueeze(1))
            next_log_probs = next_log_probs.gather(1, mu.unsqueeze(1))
            #print("log prb", next_log_probs.shape)
            #print("log prb", next_log_probs)
            q_target_1, q_target_2 = self.qnetwork_target(next_states)
            q_target_1 = q_target_1.gather(1, actions)
            q_target_2 = q_target_2.gather(1, actions)
            
            q_target_min = torch.min(q_target_1, q_target_2)
            #print(q_target)
            #print("q_target", q_target.shape)
            #print("next_log_probs", next_log_probs.shape)
            #print(q_target.shape)
            Q_targets = rewards + self.gamma * dones * (torch.sum(action_prob_next* alpha * next_log_probs - q_target_min))
            # print(Q_targets)

        Q_expected_1, Q_expected_2 = self.qnetwork_local(states)
        Q_expected_1 = Q_expected_1.gather(1, actions)
        Q_expected_2 = Q_expected_2.gather(1, actions)
        Q_expected_min = torch.min(Q_expected_1, Q_expected_2)
        
        # Compute loss
        loss = F.mse_loss(Q_expected_1, Q_targets.detach()) + F.mse_loss(Q_expected_2, Q_targets.detach())
        writer.add_scalar('Q_loss', loss, self.t_step)
        # Minimize the loss
        
        mu, pi, action_prob, log_action_prob, logits = self.actor(states.detach())
        pi_entropy = action_prob.detach() * log_action_prob.detach()
        #print(action_prob)
        #print("log", log_action_prob)
        pi_entropy  = -pi_entropy.sum(dim=-1, keepdim=True)
        # print(pi_entropy)
        # print("alph ", self.log_alpha)
        alpha_backup =  (self.target_entropy - pi_entropy).detach().mean()
        alpha_loss =  -self.log_alpha  * alpha_backup
        
        # print("loss", alpha_loss)
        # print("Alpha  {}  Alpha target {:.2f} ".format(alpha[0], self.target_entropy))
        #print("alpha", alpha)
        writer.add_scalar('A_loss', alpha_loss, self.t_step)
        
        

        # update actor -----------------------------------------------------------------------------------
        polciy_loss = (action_prob * (alpha.detach() * log_action_prob - Q_expected_min.detach())).mean()
        #print("pol los", polciy_loss)
        writer.add_scalar('P_loss', polciy_loss, self.t_step)
        
        
        
        # Minimize the critic loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # Minimize the alpha loss 
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()



        # Minimize the policy loss
        self.optimizer_actor.zero_grad()
        polciy_loss.backward()
        self.optimizer_actor.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
