# %%
import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import plotLearning_PG

# %%
class ActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def __call__(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer(object):
    def __init__(self, mem_size, input_shape, n_actions):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

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
        new_states = self.new_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminals

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        if not os.path.exists(chkpt_dir):
            os.mkdir(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')

        # compute state value V(s)
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.init_weights(self.fc1)
        self.batch_norm1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.init_weights(self.fc2)
        self.batch_norm2 = nn.LayerNorm(fc2_dims)

        # compute action value Q(s,a)
        self.fc_a = nn.Linear(n_actions, fc2_dims)
        self.init_weights(self.fc_a, f_val=0.003)

        self.fc_q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.to(self.device)
        
    
    def init_weights(self, layer, f_val=None):
        if f_val is None:
            f_val = 1./np.sqrt(layer.weight.data.size()[0])
        nn.init.uniform_(layer.weight.data, -f_val, f_val)
        nn.init.uniform_(layer.bias.data, -f_val, f_val)
    
    def forward(self, state, action):
        state_value = F.relu(self.batch_norm1(self.fc1(state)))
        state_value = self.batch_norm2(self.fc2(state_value))
        
        action_value = F.relu(self.fc_a(action))

        state_action_value = F.relu(torch.add(state_value, action_value))

        state_action_value = self.fc_q(state_action_value)

        return state_action_value
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='actor', chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        if not os.path.exists(chkpt_dir):
            os.mkdir(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.init_weights(self.fc1)
        self.batch_norm1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.init_weights(self.fc2)
        self.batch_norm2 = nn.LayerNorm(fc2_dims)

        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.init_weights(self.fc3, f_val=0.003)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.to(self.device)
        
    def init_weights(self, layer, f_val=None):
        if f_val is None:
            f_val = 1./np.sqrt(layer.weight.data.size()[0])
        nn.init.uniform_(layer.weight.data, 0, f_val)
        nn.init.uniform_(layer.bias.data, 0, f_val)
    
    def forward(self, state):
        x = F.relu(self.batch_norm1(self.fc1(state)))
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))

        return x
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent(object):
    def __init__(self, input_dims, n_actions, env, lr_actor=1e-5, lr_critic=1e-5, tau=5e-3, 
                gamma=0.99, mem_size=1e6, batch_size=64, fc1_dims=512, fc2_dims=256):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise = ActionNoise(mu=np.zeros(n_actions))
        self.memory = ReplayBuffer(mem_size=int(mem_size), input_shape=input_dims, n_actions=n_actions)
        self.actor = ActorNetwork(lr=lr_actor, input_dims=input_dims, n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='actor')
        self.critic = CriticNetwork(lr=lr_critic, input_dims=input_dims, n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='critic')
        self.target_actor = ActorNetwork(lr=lr_actor, input_dims=input_dims, n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='target_actor')
        self.target_critic = CriticNetwork(lr=lr_critic, input_dims=input_dims, n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='target_critic')

        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        # 此处说一下 model.train() 和 model.eval() 的区别
        # train() 启用 BatchNormalization 和 Dropout
        # eval() 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，pytorch框架会自动把BN和Dropout固定住，
        #        不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果。
        self.actor.eval() # 在利用原始模型进行forward之前，一定要先进行model.eval()操作，不启用 BatchNormalization 和 Dropout。
        observation = torch.Tensor(observation).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_add_noise = mu + torch.Tensor(self.noise()).to(self.actor.device)
        self.actor.train() # TODO
        return mu_add_noise.cpu().detach().numpy() 
    
    def remember(self, state, action, reward, state_new, done):
        self.memory.store_transition(state, action, reward, state_new, done)
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(states, dtype=torch.float).to(self.critic.device)
        new_states = torch.tensor(new_states, dtype=torch.float).to(self.critic.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.critic.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.critic.device)
        dones = torch.tensor(dones).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_states)
        target_critic_value_new = self.target_critic.forward(new_states, target_actions)
        critic_value = self.critic.forward(states, actions)

        targets = []
        for j in range(self.batch_size):
            targets.append(rewards[j] + self.gamma * target_critic_value_new[j] * (1-dones[j])) # vonpute target value y_i 
        targets = torch.Tensor(targets).to(self.critic.device).view(self.batch_size, 1) 

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(targets, critic_value) # update critic by minimize MSE_loss(y_i, Q(s,a))
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(states) # compute actions a of actor without noise
        self.actor.train()
        actor_loss = - self.critic.forward(states, mu) # compute Q(s,a)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters() # update target networks
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_state_dict = dict(self.actor.named_parameters()) # 给出网络层的名字和参数的迭代器
        critic_state_dict = dict(self.critic.named_parameters()) 
        target_actor_dict = dict(self.target_actor.named_parameters()) 
        target_critic_dict = dict(self.target_critic.named_parameters())

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)
    
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

# %%
if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0' #  LunarLanderContinuous-v2, MountainCarContinuous-v0
    env = gym.make(env_name)
    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]
    agent = Agent(input_dims=input_dims, n_actions=n_actions, env=env, 
            lr_actor=2.5e-5, lr_critic=2.5e-4, tau=0.001, batch_size=64, fc1_dims=400, fc2_dims=300)

    n_games = 1000
    np.random.seed(0)
    score_history = []

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_new, reward, done, _ = env.step(action)
            agent.remember(observation, action, reward, observation_new, done)
            agent.learn()

            score += reward
            observation = observation_new
        score_history.append(score)

        # if i % 25 == 0:
        #     agent.save_models()

        print('episode: {:^3d} | score: {:^10.2f} | avg_score: {:^10.2f} |'.format(i, score, np.mean(score_history[-100:])))

    filename = env_name + '_DDPG.png'
    plotLearning_PG(score_history, filename=filename, window=100)


# %%

