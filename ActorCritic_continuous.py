# %%
import os
import gym
from gym import wrappers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import plotLearning_PG

# %% 
class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(GenericNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, observation):
        x = F.relu(self.fc1(observation)) # (batch_size, fc1_dims)
        x = F.relu(self.fc2(x))     # (batch_size, fc1_dims)
        x = self.fc3(x)                   # (batch_size, n_actions)
        return x

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc_pi = nn.Linear(self.fc2_dims, self.n_actions)
        self.fc_v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, observation):
        x = F.relu(self.fc1(observation)) # (batch_size, fc1_dims)
        x = F.relu(self.fc2(x))     # (batch_size, fc1_dims)
        pi = self.fc_pi(x)                   # (batch_size)
        v = self.fc_v(x)                # (batch_size, n_actions)
        return pi, v
    
class Agent(object):
    """ Agent class for use with separate actor and critic networks.
        This is appropriate for very simple environments, such as the mountaincar
    """
    def __init__(self, gamma, lr_actor, lr_critic, input_dims, n_actions, n_outputs=1):
        self.gamma = gamma
        self.n_outputs = n_outputs

        self.actor = GenericNetwork(lr_actor, input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions*2) 
        # For the continuous mountain car, the action number actually is 1. 
        # Since this is actor for estimating policy and we use normal distribution, 
        # the n_action is set as 2 for mean and standard deviation of the normal distribution.

        self.critic = GenericNetwork(lr_critic, input_dims, fc1_dims=256, fc2_dims=256, n_actions=1)
        # For critic network, it only computes a value, so the n_output = 1.

        self.log_probs = None
    
    def choose_action(self, observation):
        # since the environment is continuous one, continous actions are derived. 
        # therefore, use Normal distribution to approximate.
        observation = torch.tensor([observation], dtype=torch.float32).to(self.actor.device)
        mu, sigma = self.actor.forward(observation)[0]
        sigma = torch.exp(sigma) # make sigma be positive.
        action_probs = torch.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=torch.Size([self.n_outputs])) # single sampled probability from normal distribution
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        action = torch.tanh(probs) # bound the action to [-1,1]

        return action.item()
    
    def learn(self, state, reward, new_state, done):

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
        new_state = torch.tensor([new_state], dtype=torch.float).to(self.actor.device)
        reward = torch.tensor([reward], dtype=torch.float).to(self.actor.device)
        terminal = torch.tensor([done], dtype=torch.bool).to(self.actor.device)
        
        critic_value_new = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)

        delta = reward + self.gamma*critic_value_new*(1-int(terminal)) - critic_value

        actor_loss = - self.log_probs*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

class CompactAgent(object):
    """ Agent class for use with a single actor critic network that shares
        the lowest layers. For use with more complex environments such as
        the discrete lunar lander
    """
    def __init__(self, gamma, lr, input_dims, n_actions, n_outputs=1):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(lr, input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions*2)

        self.n_outputs = n_outputs
        self.log_probs = None

    def choose_action(self, observation):
        # since the environment is continuous one, continous actions are derived. 
        # therefore, use Normal distribution to approximate.
        observation = torch.tensor([observation], dtype=torch.float).to(self.actor_critic.device)
        pi, v = self.actor_critic.forward(observation)
        mu, sigma = pi
        sigma = torch.exp(sigma)
        action_probs = torch.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=torch.size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        action = torch.tanh(probs)

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor_critic.optimizer.zero_grad()

        state = torch.tensor(state, dtype=torch.float32).to(self.actor_critic.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.actor_critic.device)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(self.actor_critic.device)
        terminal = torch.tensor(done, dtype=torch.bool).to(self.actor_critic.device)

        _, critic_value_ = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)

        delta = reward + self.gamma*critic_value_*(1-terminal.int()) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        

        self.actor_critic.optimizer.step()

# %%

if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0' 
    env = gym.make(env_name)
    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]
    agent = Agent(gamma=0.99, lr_actor=5e-6, lr_critic=1e-5, input_dims=input_dims, n_actions=n_actions)

    score_history = []
    score = 0
    num_episodes = 2000
    for i in range(num_episodes):
        # env = wrappers.Monitor(env, "tmp/MountainCarContinuous-v0", video_callable=lambda episode_id: True, force=True)
        done = False
        score = 0
        observation = env.reset() # np.array
        while not done:
            action = agent.choose_action(observation)
            observation_new, reward, done, _ = env.step([action])
            agent.learn(observation, reward, observation_new, done)
            observation = observation_new
            score += reward

        score_history.append(score)
        print('episode: ', i,'score: %.2f' % score)

    filename = env_name + '_ActorCrtic_continuous.png'
    plotLearning_PG(score_history, filename=filename, window=20)