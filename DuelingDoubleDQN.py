# %%
import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import plotLearning

# %%
class ReplayBuffer(object):
    def __init__(self, max_size, input_dims, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
    
    def store_transition(self, state, action, reward, state_new, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_new
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False) # acturally, they are index of batches.

        state_batch = self.state_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        return state_batch, action_batch, reward_batch, new_state_batch, terminal_batch
    
class DuelingLinearDQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/duelingdqn'):
        super(DuelingLinearDQNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc_V = nn.Linear(self.fc2_dims, 1)
        self.fc_A = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'duelling_dqn')
    
    def forward(self, state):
        x = F.relu(self.fc1(state)) # (batch_size, fc1_dims)
        x = F.relu(self.fc2(x))     # (batch_size, fc1_dims)
        V = self.fc_V(x)                   # (batch_size)
        A = self.fc_A(x)                # (batch_size, n_actions)
        return V, A
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, 
                 mem_size=100000, eps_min=0.01, eps_dec=5e-7, replace=100, chkpt_dir='tmp/duelingdoubledqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.iter_cntr = 0
        self.replace_target_cnt = replace

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.Q_eval = DuelingLinearDQNetwork(lr, n_actions=n_actions, input_dims=input_dims, name='Q_eval',
                                    fc1_dims=128, fc2_dims=128, chkpt_dir=chkpt_dir)
        self.Q_next = DuelingLinearDQNetwork(lr, n_actions=n_actions, input_dims=input_dims, name='Q_next',
                                    fc1_dims=128, fc2_dims=128, chkpt_dir=chkpt_dir)
    
    def store_transition(self, state, action, reward, state_new, done):
        self.memory.store_transition(state, action, reward, state_new, done)
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # observation = observation[np.newaxis,:] # np.newaxis : increase the dimension of the existing array by one more dimension.
            state = torch.tensor([observation], dtype=torch.float32).to(self.Q_eval.device)
            _, advantages = self.Q_eval.forward(state)
            action = torch.argmax(advantages).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def replace_target_network(self):
        if (self.replace_target_cnt is not None) and (self.iter_cntr % self.replace_target_cnt == 0):
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        self.replace_target_network()
        
        state_batch, action_batch, reward_batch, new_state_batch, terminal_batch = self.memory.sample_buffer(self.batch_size)
        state_batch = torch.tensor(state_batch).to(self.Q_eval.device)
        new_state_batch = torch.tensor(new_state_batch).to(self.Q_eval.device)
        action_batch = torch.tensor(action_batch).to(self.Q_eval.device)
        reward_batch = torch.tensor(reward_batch).to(self.Q_eval.device)
        terminal_batch = torch.tensor(terminal_batch).to(self.Q_eval.device)

        V_s, A_s = self.Q_eval.forward(state_batch)
        V_s_new, A_s_new = self.Q_next.forward(new_state_batch)  
        V_s_new_eval,  A_s_new_eval = self.Q_eval.forward(new_state_batch)

        q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True))).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)  # for action_batch
        q_next = torch.add(V_s, (A_s_new - A_s_new.mean(dim=1, keepdim=True)))  # for all actions
        q_eval = torch.add(V_s_new_eval, (A_s_new_eval - A_s_new_eval.mean(dim=1, keepdim=True))) # for all actions

        # max_actions = torch.argmax(q_eval, dim=1)
        q_next[terminal_batch] = 0.0  
        q_target = reward_batch + self.gamma * torch.max(q_eval, dim=1)[0].detach()

        # loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.iter_cntr += 1
        self.decrement_epsilon()


    def save_models(self):
        self.Q_eval.save_checkpoint()
        self.Q_next.save_checkpoint()
    
    def load_models(self):
        self.Q_eval.load_checkpoint()
        self.Q_next.load_checkpoint()

# %%
if __name__ == '__main__':
    env_name = 'CartPole-v1' # CartPole-v1, LunarLander-v2, .......
    env = gym.make(env_name)
    input_dims = env.observation_space.shape
    n_actions = env.action_space.n
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=n_actions, eps_min=0.01, eps_dec=1e-4,
                  input_dims=input_dims, lr=0.001, replace=100, mem_size=100000)
    n_games = 500
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
    
    scores, eps_hist = [], []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_new, reward, done, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_new, done) 
            agent.learn()
            observation = observation_new

        scores.append(score)
        eps_hist.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print('episode', i, 'score %.2f' % score, 'average_score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
        # if i > 0 and i % 10 == 0:
        #    agent.save_models()
    x = [i+1 for i in range(n_games)]
    filename = env_name + '_DuelingDoubleDQN.png'
    plotLearning(x, scores, eps_hist, filename)  