# %%
import os
import gym
import pybullet_envs
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import plotLearning_PG

# %%
class ReplayBuffer():
    def __init__(self, mem_size, input_shape, n_actions):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_new, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_new
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_new = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_new, dones

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, 
                name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        if not os.path.exists(chkpt_dir):
            os.mkdir(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')

        # compute state value V(s)
        self.fc1 = nn.Linear(input_dims+n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc_q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state, action):
        action_value = F.relu(self.fc1(torch.cat([state,action], dim=1)))
        action_value = F.relu(self.fc2(action_value))
        q = self.fc_q(action_value)

        return q
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=256, fc2_dims=256,
                name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        if not os.path.exists(chkpt_dir):
            os.mkdir(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')

        # compute state value V(s)
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc_v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))
        v = self.fc_v(state_value)
        return v
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, max_action, n_actions=2, 
                 fc1_dims=256, fc2_dims=256, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        if not os.path.exists(chkpt_dir):
            os.mkdir(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc_mu = nn.Linear(fc2_dims, n_actions)
        self.fc_sigma = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = torch.distributions.Normal(mu, sigma)
        if reparameterize:
            action_ = probabilities.rsample()
        else:
            action_ = probabilities.sample()
    
        action = torch.tanh(action_)*torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(action_)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent():
    def __init__(self, input_dims, env, n_actions,
                lr_actor=3e-4, lr_critic=3e-4, tau=0.005, reward_scale=2,
                gamma=0.99, mem_size=1e6, batch_size=256, fc1_dims=256, fc2_dims=256):
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_size=int(mem_size), input_shape=input_dims, n_actions=n_actions)

        self.actor = ActorNetwork(lr=lr_actor, input_dims=input_dims, max_action=env.action_space.high,
                            n_actions=n_actions, name='actor')
        self.critic_1 = CriticNetwork(lr=lr_critic, input_dims=input_dims, n_actions=n_actions, 
                            name='critic_1')
        self.critic_2 = CriticNetwork(lr=lr_critic, input_dims=input_dims, n_actions=n_actions, 
                            name='critic_2')
        self.value = ValueNetwork(lr=lr_critic, input_dims=input_dims, name='value')
        self.target_value = ValueNetwork(lr=lr_critic, input_dims=input_dims, name='target_value')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, state_new, done):
        self.memory.store_transition(state, action, reward, state_new, done)
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        value_state_dict = dict(self.value.named_parameters()) # 给出网络层的名字和参数的迭代器
        target_value_state_dict = dict(self.target_value.named_parameters()) 

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                                    (1-tau)*target_value_state_dict[name].clone()
        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, states_new, dones = self.memory.sample_buffer(self.batch_size)

        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        states_new = torch.tensor(states_new, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones, dtype=torch.int64).to(self.actor.device)

        values = self.value(states).view(-1)
        values_new = self.target_value(states_new).view(-1)
        values_new[dones] = 0.0

        actions_sample, log_probs = self.actor.sample_normal(states, reparameterize=False)
        # actions_sample = torch.tensor(actions_sample, dtype=torch.float).to(self.actor.device)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(states, actions_sample)
        q2_new_policy = self.critic_2.forward(states, actions_sample)
        critic_value = torch.min(q1_new_policy, q2_new_policy).view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(values, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions_sample, log_probs = self.actor.sample_normal(states, reparameterize=True)
        # actions_sample = torch.tensor(actions_sample, dtype=torch.float).to(self.actor.device)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(states, actions_sample)
        q2_new_policy = self.critic_2.forward(states, actions_sample)
        critic_value = torch.min(q1_new_policy, q2_new_policy).view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.reward_scale*reward + self.gamma*values_new
        q1_old_policy = self.critic_1.forward(states, actions).view(-1)
        q2_old_policy = self.critic_2.forward(states, actions).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

# %%
if __name__ == '__main__':
    env_name = 'InvertedPendulumBulletEnv-v0' #  LunarLanderContinuous-v2, MountainCarContinuous-v0, InvertedPendulumBulletEnv-v0
    env = gym.make(env_name)
    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = Agent(input_dims=input_dims, n_actions=n_actions, env=env)
    n_games = 250

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
    
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_new, reward, done, _ = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_new, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_new
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        
        print('episode: {:^3d} | score: {:^10.2f} | avg_score: {:^10.2f} |'.format(i, score, avg_score))

    if not load_checkpoint:
        filename = env_name + '_SAC.png'
        plotLearning_PG(score_history, filename)
