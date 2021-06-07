# %%
import os
import gym
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from utils import plotLearning_PG

# %% 
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.99), eps=1e-8, weight_decay=0): 
        # betas are coefficients used for computing running averages of gradient and its square
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # param_groups is a dict with all params
        for group in self.param_groups: # len(param_groups) = 1
            for p in group['params']: # len(group['params']) = 8
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Once the tensor/storage is moved to shared_memory, it will be possible to send it to other processes without making any copies.
                state['exp_avg'].share_memory_() 
                state['exp_avg_sq'].share_memory_()

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=128, fc2_dims=128):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(*input_dims, fc2_dims)
        self.fc_pi = nn.Linear(fc1_dims, n_actions)
        self.fc_v = nn.Linear(fc2_dims, 1)
        # for layer in [self.fc1,self.fc2, self.fc_pi, self.fc_v]:
        #     nn.init.normal_(layer.weight, mean=0., std=0.1)
        #     nn.init.constant_(layer.bias, 0.)
    
    def forward(self, observation):
        x1 = torch.tanh(self.fc1(observation))
        x2 = torch.tanh(self.fc2(observation))
        pi = self.fc_pi(x1)                   # (batch_size)
        v = self.fc_v(x2)                # (batch_size, n_actions)
        return pi, v
    
class Agent():
    def __init__(self, input_dims, n_actions, gamma=0.99):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(input_dims=input_dims, n_actions=n_actions, fc1_dims=128, fc2_dims=128)
    
    def calc_return(self, states, rewards, done):
        _, v = self.actor_critic.forward(states)

        R = v[-1]*(1-int(done))

        return_buffer = []
        for reward in rewards[::-1]:
            R = reward + self.gamma*R
            return_buffer.append(R)
        return_buffer.reverse()
        return_buffer = torch.tensor(return_buffer, dtype=torch.float)
        return return_buffer
    
    def calc_loss(self, states, actions, rewards, done):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)

        returns = self.calc_return(states, rewards, done)

        pi, values = self.actor_critic.forward(states)
        critic_loss = (returns-values.squeeze())**2

        probabilities = torch.softmax(pi, dim=1)
        distribution = torch.distributions.Categorical(probabilities)
        log_probs = distribution.log_prob(actions)
        actor_loss = -log_probs*(returns-values.squeeze())

        total_loss = (critic_loss + actor_loss).mean()
        return total_loss
    
    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float)
        pi, v = self.actor_critic.forward(state)
        probs = torch.softmax(pi, dim=1)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample().numpy()[0]

        return action
    
class Worker(mp.Process):
    """ A3C class for use with separate actor and critic networks.
        This is appropriate for very simple environments, such as the mountaincar
    """
    def __init__(self, env, global_actor_critic, optimizer, global_episode_idx, score_queue, input_dims, n_actions, name, gamma=0.99):
        super(Worker, self).__init__()
        self.env = env
        self.global_actor_critic = global_actor_critic
        self.optimizer = optimizer

        self.local_actor_critic = Agent(input_dims=input_dims, n_actions=n_actions, gamma=gamma)
        
        self.name = 'w%02i' % name
        self.global_episode_idx = global_episode_idx
        self.score_queue = score_queue
    
    def run(self):
        total_step = 1
        while self.global_episode_idx.value < N_GAMES:
            done = False
            observation = self.env.reset()
            score = 0
            states_buffer, rewards_buffer, actions_buffer = [], [], []

            while True:
                action = self.local_actor_critic.choose_action(observation)
                observation_new, reward, done, _ = self.env.step(action)
                score += reward

                states_buffer.append(observation)
                rewards_buffer.append(reward)
                actions_buffer.append(action)

                if total_step % T_MAX == 0 or done:

                    loss = self.local_actor_critic.calc_loss(states_buffer, actions_buffer, rewards_buffer, done)
                    self.optimizer.zero_grad()
                    loss.backward()

                    for local_param, global_param in zip(self.local_actor_critic.actor_critic.parameters(), 
                                                         self.global_actor_critic.actor_critic.parameters()):
                        global_param._grad = local_param.grad # share the local_param to global model

                    self.optimizer.step()

                    self.local_actor_critic.actor_critic.load_state_dict(self.global_actor_critic.actor_critic.state_dict()) # share updated glocal networks' parameters to local model
                    
                    states_buffer, rewards_buffer, actions_buffer = [], [], []

                    if done:
                        # record the global episode index and reward 
                        with self.global_episode_idx.get_lock():
                            self.global_episode_idx.value += 1
                        score_queue.put(score)
                        break
                        
                total_step += 1
                observation = observation_new
            print(self.name, 'episode: ', self.global_episode_idx.value, 'score: %.2f' % score)
        self.score_queue.put(None)

# %%
if __name__ == '__main__':
    env_name = 'CartPole-v1' # CartPole-v1, LunarLander-v2, .......
    env = gym.make(env_name).unwrapped
    input_dims = env.observation_space.shape
    n_actions = env.action_space.n
    lr = 1e-4
    N_GAMES = 3000
    T_MAX = 5
    GAMMA = 0.9

    global_actor_critic = Agent(input_dims=input_dims, n_actions=n_actions, gamma=GAMMA) # global network
    global_actor_critic.actor_critic.share_memory() # share the global parameters in multiprocessing
    optimizer = SharedAdam(global_actor_critic.actor_critic.parameters(), lr=lr, betas=(0.92,0.999)) # global optimizer
    global_episode_idx = mp.Value('i', 0)
    global_episode_reward = mp.Value('d', 0.)
    score_queue = mp.Queue()

    # parallel training
    workers = [Worker(env=env, 
                      global_actor_critic=global_actor_critic, 
                      optimizer=optimizer, 
                      input_dims=input_dims, 
                      n_actions=n_actions, 
                      name=i, 
                      global_episode_idx=global_episode_idx,
                      score_queue=score_queue,
                      gamma=GAMMA) 
               for i in range(mp.cpu_count())]

    [worker.start() for worker in workers]
    scores = []
    while True:
        score = score_queue.get()
        if score is not None:
            scores.append(score)
        else:
            break
    [worker.join() for worker in workers]

    filename = env_name + '_A3C_discrete.png'
    plotLearning_PG(scores, filename=filename, window=20)