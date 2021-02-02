import numpy as np
import sys
import random
from collections import namedtuple, deque

from model import QNetwork, Actor

import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import GumbelSoftmax
from gym import wrappers

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
        print("seed", self.seed)
        # Q-Network
        torch.manual_seed(self.seed)
        np.random.seed(seed=self.seed)
        random.seed(self.seed)
        self.target_entropy = config["target_entropy"]
        # self.target_entropy = 1.358
        self.target_entropy = -4  # 1.358 
        self.actor = Actor(state_size, action_size, self.seed, config["clip"]).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.qnetwork_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=config["device"])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.alpha = self.log_alpha.detach().exp()
        self.alpha = 0.0
        self.t_step = 0
        self.update_freq = config["update_freq"]
        self.vid_path = config["locexp"] + "/vid"
        self.action_pd = GumbelSoftmax
    
    def step(self, memory, writer):
        self.t_step += 1 
        if self.t_step > 1000:
            if self.t_step % self.update_freq == 0:
                self.learn(memory, writer)
    
    def act_dqn(self, state, eps=0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.Q1(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def act(self, state, eps):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(state)
        self.actor.train()
        action_pd = self.get_policy(logits)
        log_probs, action = self.calc_log_prob_action(action_pd)
        #print(next_actions)
        if random.random() > eps:
            return action.item()
        else:
            return random.choice(np.arange(self.action_size))

    
    def guard_actions(self, actions):
        actions = F.one_hot(actions.long(), self.action_size).float()
        return actions
    
    def learn(self, memory, writer):
        states, actions, rewards, next_states, dones = memory.sample(self.batch_size)
        with torch.no_grad():
            logits_next = self.actor(next_states)
            action_pd = self.get_policy(logits_next)
            log_probs_next, next_actions = self.calc_log_prob_action(action_pd)
            next_actions = self.guard_actions(next_actions)
            qn1, qn2 = self.qnetwork_target(next_states, next_actions)
            Q_targets = torch.min(qn1, qn2)
            
            Q_targets = (Q_targets - self.alpha * log_probs_next.unsqueeze(1))
            Q_targets_next = rewards + (self.gamma * Q_targets * dones)
            
        actions = self.guard_actions(actions.squeeze(1))
        pred_q1, pred_q2 = self.qnetwork_local(states, actions)
        q_loss = F.mse_loss(Q_targets_next, pred_q1) +  F.mse_loss(Q_targets_next, pred_q2) 
        # -----------------------update-q-network------------------------------------------------------------
        self.optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 0.5)
        self.optimizer.step()
        writer.add_scalar('loss/critic', q_loss, self.t_step)
        
        # -----------------------policy-loss------------------------------------------------------------
        
        logits = self.actor(states) 
        action_pd = self.get_policy(logits)
        log_probs, reparam_actions = self.calc_log_prob_action(action_pd, True)
        #print(prob_reparam_actions)
        #print(prob_reparam_actions.shape)
        pred_q1, pred_q2 = self.qnetwork_local(states, reparam_actions)
        q_pred = torch.min(pred_q1, pred_q2)
        #entropy = -torch.sum(prob_reparam_actions * log_probs, dim=1, keepdim=True)
        policy_loss = (self.alpha * log_probs - q_pred).mean()
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        writer.add_scalar('loss/policy', policy_loss, self.t_step)
        self.optimizer_actor.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        # -----------------------entropy-loss------------------------------------------------------------
        entropy_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        entropy_loss.backward()
        writer.add_scalar('loss/alpha', entropy_loss, self.t_step)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.detach().exp()
        writer.add_scalar('alpha', self.alpha, self.t_step)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

                

    def get_policy(self, pdparam):
        pd_kwargs = {'temperature': torch.tensor(1.0)}
        action_pd = self.action_pd(logits=pdparam, **pd_kwargs)
        return action_pd

    def calc_log_prob_action(self, action_pd, reparam=False):
        '''Calculate log_probs and actions with option to reparametrize from paper eq. 11'''
        actions = action_pd.rsample() if reparam else action_pd.sample()
        log_probs = action_pd.log_prob(actions)
        return log_probs, actions

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
    
    def eval_policy(self, env, writer, eval_episodes=4):
        env  = wrappers.Monitor(env, str(self.vid_path) + "/{}".format(self.t_step), video_callable=lambda episode_id: True,force=True)
        average_reward = 0
        scores_window = deque(maxlen=eval_episodes)
        for i_epiosde in range(eval_episodes):
            print("Eval Episode {} of {} ".format(i_epiosde, eval_episodes))
            episode_reward = 0
            state = env.reset()
            while True:
                action = self.act(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    break
            scores_window.append(episode_reward)
        average_reward = np.mean(scores_window)
        writer.add_scalar('Eval_reward', average_reward, self.t_step)

