# %% 
import os
import gym
from gym import wrappers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import plotLearning_PG

# %%
class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(PolicyNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state): # reminder here, state will be in cuda
        x = F.relu(self.fc1(state)) # (batch_size, fc1_dims)
        x = F.relu(self.fc2(x))     # (batch_size, fc1_dims)
        x = self.fc3(x)                   # (batch_size)
        return x
    
class Agent(object):
    def __init__(self, gamma, lr, input_dims, n_actions):
        self.gamma = gamma
        self.lr = lr
        self.iter_cntr = 0

        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(lr, n_actions=n_actions, input_dims=input_dims, 
                                    fc1_dims=256, fc2_dims=256)
    
    def choose_action(self, observation):
        observation = torch.tensor([observation], dtype=torch.float32).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(observation))  # probabilities of each actioon in the action space. Sum of the probabilities = 1
        action_probs = torch.distributions.Categorical(probabilities)  # --> 形成categorical distribution
        action = action_probs.sample() # 按概率sample action

        log_probs = action_probs.log_prob(action) # --> loss function 
        self.action_memory.append(log_probs) # log likelihood

        return action.item()
    
    def store_reward(self, reward):
        self.reward_memory.append(reward)
    
    def learn(self):
        self.policy.optimizer.zero_grad()

        G = np.zeros_like(self.reward_memory, dtype=np.float32)
        for t in range(len(self.reward_memory)): # for each episode
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)): # for each step in the episode
                G_sum += discount * self.reward_memory[k]
                discount *= self.gamma
            G[t] = G_sum # 第t步到最后的expected cumulative reward
        mean = np.mean(G)
        std = np.std(G) if np.std(G)>0 else 1
        G = (G - mean) / std

        G = torch.tensor(G, dtype=torch.float32).to(self.policy.device)

        loss = 0
        for g, log_prob in zip(G, self.action_memory):
            loss += -g * log_prob
        
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []

# %%
if __name__ == '__main__':
    env_name = 'LunarLander-v2' # CartPole-v1, LunarLander-v2, .......
    env = gym.make(env_name)
    input_dims = env.observation_space.shape
    n_actions = env.action_space.n
    agent = Agent(gamma=0.99, n_actions=n_actions, input_dims=input_dims, lr=0.0002)
    
    score_hist = []
    score = 0
    num_episodes = 2500

    #env = wrappers.Monitor(env, "tmp/lunar-lander", video_callable=lambda episode_id: True, force=True)

    for i in range(num_episodes):
        print('episode: ', i, ' score: ', score)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_new, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            observation = observation_new
            score += reward
        
        score_hist.append(score)
        agent.learn()
    
    filename = env_name + '_PG.png'
    plotLearning_PG(score_hist, filename=filename, window=25)