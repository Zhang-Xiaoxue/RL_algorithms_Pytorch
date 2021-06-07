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
    def __init__(self, gamma, lr_actor, lr_critic, input_dims, n_actions):
        self.gamma = gamma

        self.actor = ActorCriticNetwork(lr_actor, input_dims, fc1_dims=128, fc2_dims=128, n_actions=n_actions)

        self.critic = ActorCriticNetwork(lr_critic, input_dims, fc1_dims=128, fc2_dims=128, n_actions=1)

        self.log_probs = None
    
    def choose_action(self, observation):
        observation = torch.tensor([observation], dtype=torch.float32).to(self.actor_critic.device)
        probabilities, _ = self.actor.forward(observation)
        probabilities = F.softmax(probabilities)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)

        return action.item()
    
    def learn(self, state, reward, new_state, done):

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        state = torch.tensor(state).to(self.actor.device)
        new_state = torch.tensor(new_state).to(self.actor.device)
        reward = torch.tensor(reward).to(self.actor.device)
        terminal = torch.tensor(done).to(self.actor.device)
        
        critic_value_new = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)

        delta = reward + self.gamma*critic_value_new*(1-terminal.int()) - critic_value

        actor_loss = - self.log_probsh*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

class CompactAgent(object):
    """ Agent class for use with a single actor critic network that shares
        the lowest layers. For use with more complex environments such as
        the discrete lunar lander
    """
    def __init__(self, gamma, lr, input_dims, n_actions):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(lr, input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions)

        self.log_probs = None

    def choose_action(self, observation):
        observation = torch.tensor([observation], dtype=torch.float32).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor_critic.optimizer.zero_grad()

        state = torch.tensor(state, dtype=torch.float).to(self.actor_critic.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor_critic.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.actor_critic.device)
        terminal = torch.tensor(done, dtype=torch.float).to(self.actor_critic.device)

        _, critic_value_ = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)

        delta = reward + self.gamma*critic_value_*(1-terminal.int()) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        

        self.actor_critic.optimizer.step()

# %%
if __name__ == '__main__':
    env_name = 'LunarLander-v2' # CartPole-v1, LunarLander-v2, .......
    env = gym.make(env_name)
    input_dims = env.observation_space.shape
    n_actions = env.action_space.n
    agent = CompactAgent(gamma=0.99, lr=1e-5, input_dims=input_dims, n_actions=n_actions)

    score_history = []
    score = 0
    num_episodes = 1000
    for i in range(num_episodes):
        # env = wrappers.Monitor(env, "tmp/lunar-lander", video_callable=lambda episode_id: True, force=True)
        done = False
        score = 0
        observation = env.reset() # np.array
        while not done:
            action = agent.choose_action(observation)
            observation_new, reward, done, _ = env.step(action)
            agent.learn(observation, reward, observation_new, done)
            observation = observation_new
            score += reward

        score_history.append(score)
        print('episode: ', i,'score: %.2f' % score)
        # print('episode: ', i,'score: %.2f' % score, 'Param Mean: %.8e' %torch.mean(agent.actor_critic.state_dict()['fc_pi.weight']).item())

    filename = env_name + '_ActorCrtic_discrete.png'
    plotLearning_PG(score_history, filename=filename, window=50)
    