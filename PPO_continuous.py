# %%
import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from utils import plotLearning_PG

# %%
class memory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
            np.array(self.values), np.array(self.rewards), np.array(self.dones), batches
    
    def store_memory(self, state, action, probs, values, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        ####### Different from PPO_discrete
        mu, sigma = self.actor(state)[0]
        sigma = torch.exp(sigma)
        distribution = torch.distributions.Normal(mu, sigma)
        return distribution
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        torch.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo_continuous'):
        super(CriticNetwork, self).__init__()
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        value = self.critic(state)
        return value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        torch.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=0.0003, gae_lambda=0.95, 
                 policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions=n_actions*2, input_dims=input_dims, lr=lr)
        self.critic = CriticNetwork(input_dims=input_dims, lr=lr)
        self.memory = memory(batch_size=batch_size)
    
    def remember(self, state, action, probs, values, reward, done):
        self.memory.store_memory(state, action, probs, values, reward, done)
    
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
    
    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, values_arr, reward_arr, done_arr, batches = self.memory.generate_batches()

            values = values_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(done_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda

                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch], dtype=torch.float).to(self.actor.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.float).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()
                # critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = - torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value).pow(2).mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

# %%
if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0' # 
    env = gym.make(env_name)
    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]
    N = 50
    batch_size = 10
    n_epochs = 10
    lr = 1e-4
    agent = Agent(n_actions=n_actions, input_dims=input_dims, batch_size=batch_size, lr=lr, n_epochs=n_epochs)

    n_games = 500

    best_score = env.reward_range[0]
    score_history = []
    avg_score = 0


    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        n_steps, learn_iters = 0, 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_new, reward, done, _ = env.step([action])

            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_new
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        print('episode: {:^3d} | score: {:^10.2f} | avg_score: {:^10.2f} | time_steps: {:^5d} | learning_steps: {:^3d} |'.format(i, score, avg_score, n_steps, learn_iters))

    x = [i+1 for i in range(len(score_history))]
    filename = env_name + '_PPO_continuous.png'
    plotLearning_PG(score_history, filename=filename, x=x)
