# %%
import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import plotLearning_PG

# %% 
class ReplayBuffer(object): # for continous action
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.log_prob_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
    
    def store_transition(self, state, log_prob, reward, state_new, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_new
        self.reward_memory[index] = reward
        self.log_prob_memory[index] = log_prob
        self.terminal_memory[index] = done

        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False) # acturally, they are index of batches.

        state_batch = self.state_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        log_prob_batch = self.log_prob_memory[batch]
        reward_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        return state_batch, log_prob_batch, reward_batch, new_state_batch, terminal_batch

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
    
    def forward(self, state):
        x = F.relu(self.fc1(state)) # (batch_size, fc1_dims)
        x = F.relu(self.fc2(x))     # (batch_size, fc1_dims)
        pi = self.fc_pi(x)                   # (batch_size)
        v = self.fc_v(x)                # (batch_size, n_actions)
        return pi, v
    
class Agent(object):
    def __init__(self, gamma, lr, input_dims, n_actions, batch_size=32, mem_size=100000):
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size

        self.memory = ReplayBuffer(mem_size, input_dims)
        self.actor_critic = ActorCriticNetwork(lr, input_dims, fc1_dims=128, fc2_dims=128, n_actions=n_actions)
    
    def store_transition(self, state, action, reward, state_new, done):
        self.memory.store_transition(state, action, reward, state_new, done)
    
    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float32).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities = F.softmax(probabilities)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)

        return action.item(), log_probs
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.actor_critic.optimizer.zero_grad()
        
        state_batch, log_prob_batch, reward_batch, new_state_batch, terminal_batch = self.memory.sample_buffer(self.batch_size)
        state_batch = torch.tensor(state_batch).to(self.actor_critic.device)
        new_state_batch = torch.tensor(new_state_batch).to(self.actor_critic.device)
        log_prob_batch = torch.tensor(log_prob_batch).to(self.actor_critic.device)
        reward_batch = torch.tensor(reward_batch).to(self.actor_critic.device)
        terminal_batch = torch.tensor(terminal_batch).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state_batch)
        _, critic_value_new = self.actor_critic.forward(new_state_batch)  

        critic_value_new[terminal_batch] = 0.0

        delta = reward_batch + self.gamma*critic_value_new

        actor_loss = - torch.mean(log_prob_batch*(delta - critic_value))
        critic_loss =  F.mse_loss(delta, critic_value)

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()

# %%
if __name__ == '__main__':
    env_name = 'LunarLander-v2' # CartPole-v1, LunarLander-v2, .......
    env = gym.make(env_name)
    input_dims = env.observation_space.shape
    n_actions = env.action_space.n
    agent = Agent(gamma=0.99, lr=1e-4, batch_size=64, n_actions=n_actions,
                  input_dims=input_dims, mem_size=100000)
    n_games = 1500
    
    scores = []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action, prob = agent.choose_action(observation)
            observation_new, reward, done, _ = env.step(action)
            score += reward
            agent.store_transition(observation, prob, reward, observation_new, done) 
            agent.learn()
            observation = observation_new

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('episode', i, 'score %.2f' % score, 'average_score %.2f' % avg_score)

    x = [i+1 for i in range(n_games)]
    filename = env_name + '_BadACwithReplay.png'
    plotLearning_PG(scores, filename)