#!/usr/bin/env python
# coding: utf-8

# In[1]:


import inspect
import time
from statistics import mean, stdev
import numpy as np
import os 
from datetime import datetime
import csv

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Simulator.Scenarios import DroneSwarmScenarioGenerator
from datetime import datetime
from CybORG.Agents.Wrappers import PettingZooParallelWrapper

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal, Categorical


# In[2]:


# experimental
def remove_elements_from_dict_values(input_dict, num_drones):
    n = num_drones  # number of entries in the dict
    indices_to_remove = []

    # add indices based on the provided rules
#     indices_to_remove.extend([2*n + 2, 2*n + 3])
#     indices_to_remove.extend([(2*n + 4) + 4*i for i in range(n - 1)])
#     indices_to_remove.extend([(2*n + 5) + 4*i for i in range(n - 1)])
#     indices_to_remove.extend([(2*n + 6) + 4*i for i in range(n - 1)])
    
    return {key: np.delete(value, indices_to_remove) for key, value in input_dict.items()}


# In[3]:


device = 'cpu'
num_drones = 3

hyperparameters = {
            'timesteps_per_batch': 2048, 
            'max_timesteps_per_episode': 501, 
            'gamma': 0.99, 
            'n_updates_per_iteration': 10,
            'lr': 3e-4, 
            'clip': 0.2,
            'render': True,
            'render_every_i': 10
          }


# In[4]:


class PPO:
    """
        This is the PPO class we will use as our model in main.py
    """
    def __init__(self, policy_class_actor, policy_class_critic, env, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        # Make sure the environment is compatible with our code
#         print(type(env.observation_space(f"blue_agent_0")))	# <class 'gym.spaces.multi_discrete.MultiDiscrete'>
#         print(type(env.action_space(f"blue_agent_0")))	# <class 'gym.spaces.discrete.Discrete'>

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space(f"blue_agent_0").shape[0]
        self.act_dim = len(env.get_action_space(f'blue_agent_0'))

#         print("obs, act dim = ", self.obs_dim, self.act_dim)

        # Initialize actor and critic networks
        self.actor = policy_class_actor(self.obs_dim, self.act_dim)                                                   # ALG STEP 1
        self.critic = policy_class_critic(self.obs_dim, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=hyperparameters['lr'])
        self.critic_optim = Adam(self.critic.parameters(), lr=hyperparameters['lr'])

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }


# In[5]:


class FeedForwardNN_Actor(nn.Module):
    """
        A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """
    def __init__(self, in_dim, out_dim):
        """
            Initialize the network and set up the layers.

            Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int

            Return:
                None
        """
        super(FeedForwardNN_Actor, self).__init__()

        self.layer1 = nn.Linear(in_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, out_dim)

    def forward(self, obs):
        """
            Runs a forward pass on the neural network.

            Parameters:
                obs - observation to pass as input

            Return:
                output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
#         if isinstance(obs, np.ndarray):
#             obs = torch.tensor(obs, dtype=torch.float)
        if not torch.is_tensor(obs) or obs.dtype != torch.float32:
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = self.layer3(activation2)
        output = F.softmax(activation3, dim=-1)
#         dist = Categorical(output)
#         action = dist.sample()
#         action_logprob = dist.log_prob(action)

#         return action.detach().item(), action_logprob.detach().item()
        return output

class FeedForwardNN_Critic(nn.Module):
    """
        A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """
    def __init__(self, in_dim, out_dim):
        """
            Initialize the network and set up the layers.

            Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int

            Return:
                None
        """
        super(FeedForwardNN_Critic, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        """
            Runs a forward pass on the neural network.

            Parameters:
                obs - observation to pass as input

            Return:
                output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
#         if isinstance(obs, np.ndarray):
#             obs = torch.tensor(obs, dtype=torch.float)
        if not torch.is_tensor(obs) or obs.dtype != torch.float32:
            obs = torch.tensor(obs, dtype=torch.float)

#         print('inside critic')
#         print(type(obs))
#         print(obs.shape)
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output


# In[6]:


def compute_rtgs(batch_rews, hyperparameters):

#     print(len(batch_rews))
#     print(len(batch_rews[0]),len(batch_rews[1]))
    
    returns = {}
    
    for agent, rews in batch_rews.items():
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews[agent]):

            discounted_reward = 0 # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * hyperparameters['gamma']
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        
        if agent not in returns:
            returns[agent] = []
        returns[agent].append(batch_rtgs)

#     print(returns)
    return returns


# In[7]:


def rollout(env, models, hyperparameters):
    batch_obs = {}
    batch_acts = {}
    batch_log_probs = {}
    batch_rews = {}
    batch_rtgs = []
    batch_lens = []
    ep_rews = {}
    t = 0 # Keeps track of how many timesteps we've run so far this batch

    while t < hyperparameters['timesteps_per_batch']:
        ep_rews = {} # rewards collected per episode
        obs = env.reset()
        done = False

        # Run an episode for a maximum of max_timesteps_per_episode timesteps
        for ep_t in range(hyperparameters['max_timesteps_per_episode']):
#             print(ep_t)
#             # If render is specified, render the environment
#             if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
#                 self.env.render()
            t += 1 # Increment timesteps ran this batch so far
            
#             # Track observations in this batch
#             for agent, observation in obs.items():
#                 if agent not in batch_obs:
#                     batch_obs[agent] = []
#                 print(observation)
#                 print(obs[agent])
#                 batch_obs[agent].append(np.array([observation], dtype=np.float32))

#             print(batch_obs)
            # Calculate action and make a step in the env. 
            # Note that rew is short for reward.
#             print(obs)
#             print(f'{ep_t} : obs = {obs}')
            action, log_prob = get_action(obs=obs, models=models, env=env)
#             print(action)
            # Track recent reward, action, and action log probability
            for agent, act_val in action.items():
                if agent not in batch_acts:
                    batch_acts[agent] = []
                batch_acts[agent].append(np.array([act_val], dtype=np.float32))
                if agent not in batch_obs:
                    batch_obs[agent] = []
                batch_obs[agent].append(np.array(obs[agent], dtype=np.float32))

            obs, rew, done, info = env.step(action)    
#             print(f'{ep_t} : action = {action}')
#             print(obs, rew, done)
#             print(done)
#             print('action, log_prob, obs, rew')
#             print(action)
#             print(log_prob)
#             print(obs)
#             print(rew)

            for agent, act_val in action.items():
                if agent not in batch_log_probs:
                    batch_log_probs[agent] = []
                batch_log_probs[agent].append(np.array(log_prob[agent], dtype=np.float32))
                if agent not in ep_rews:
                    ep_rews[agent] = []
                ep_rews[agent].append(rew[agent])

#             for agent, log_prob_val in log_prob.items():
#                 if agent not in batch_log_probs:
#                     batch_log_probs[agent] = []
#                 batch_log_probs[agent].append(np.array([log_prob_val], dtype=np.float32))

#             for agent, agent_rew in rew.items():
#                 if agent not in ep_rews:
#                     ep_rews[agent] = []
#                 ep_rews[agent].append(agent_rew)
                                                
            # If the environment tells us the episode is terminated, break
            if all(done.values()):
#                 print('done break')
                break

#             print("ep_t = ", ep_t)
#             print(len(batch_obs['blue_agent_0']), len(batch_acts['blue_agent_0']))
#             print(len(batch_obs['blue_agent_1']), len(batch_acts['blue_agent_1']))
#             print(len(batch_obs['blue_agent_2']), len(batch_acts['blue_agent_2']))

#         print('batch_obs, batch_acts, batch_log_probs, ep_rews')
#         print(batch_obs)
#         print(batch_acts)
#         print(batch_log_probs)
#         print(ep_rews)
        # Track episodic lengths and rewards
#         print(len(batch_obs['blue_agent_0']), len(batch_acts['blue_agent_0']))
#         print(len(batch_obs['blue_agent_1']), len(batch_acts['blue_agent_1']))
#         print(len(batch_obs['blue_agent_2']), len(batch_acts['blue_agent_2']))

        batch_lens.append(ep_t + 1)
        
#         print(len(ep_rews))
        for agent, agent_ep_rews in ep_rews.items():
            if agent not in batch_rews:
                batch_rews[agent] = []
            batch_rews[agent].append(agent_ep_rews)

#     print(batch_lens)
#     print(batch_obs)
    # Reshape data as tensors in the shape specified in function description, before returning
#     print(batch_obs)
#     batch_obs = torch.tensor(batch_obs, dtype=torch.float)
    # Convert numpy arrays to torch tensors

#     print(len(batch_obs['blue_agent_0']), len(batch_acts['blue_agent_0']))
#     print(len(batch_obs['blue_agent_1']), len(batch_acts['blue_agent_1']))
#     print(len(batch_obs['blue_agent_2']), len(batch_acts['blue_agent_2']))
    
    for key in batch_obs:
        batch_obs[key] = torch.stack([torch.from_numpy(arr) for arr in batch_obs[key]])

    for key in batch_acts:
        batch_acts[key] = torch.stack([torch.from_numpy(arr) for arr in batch_acts[key]])

#     print(len(batch_obs['blue_agent_0']), len(batch_acts['blue_agent_0']))
#     print(len(batch_obs['blue_agent_1']), len(batch_acts['blue_agent_1']))
#     print(len(batch_obs['blue_agent_2']), len(batch_acts['blue_agent_2']))

    for key in batch_log_probs:
        batch_log_probs[key] = torch.stack([torch.from_numpy(arr) for arr in batch_log_probs[key]])

#     print(batch_rews)
    batch_rtgs = compute_rtgs(batch_rews, hyperparameters)                                                              # ALG STEP 4

    # Log the episodic returns and episodic lengths in this batch.
    
    for key in batch_rews:
        models[key].logger['batch_rews'] = batch_rews[key]
        models[key].logger['batch_lens'] = batch_lens

    return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


# In[8]:


def get_action(obs, models, env):
    """
        Queries an action from the actor network, should be called from rollout.

        Parameters:
            obs - the observation at the current timestep

        Return:
            action - the action to take, as a numpy array
            log_prob - the log probability of the selected action in the distribution
    """
    actions = {}
    log_probs = {}
    
    output = {agent_name: agent.actor(obs[agent_name]) for agent_name, agent in models.items() if agent_name in env.agents} 
#     print(output)
    for key, value in output.items():
        m = Categorical(value)
        sample = m.sample()
        actions[key] = sample.item()
        log_probs[key] = m.log_prob(sample).item()
    
#     dist = Categorical(output)
#     action = dist.sample()
#     action_logprob = dist.log_prob(action)
#     print(actions_and_logprob)
    
#     actions = {agent: value[0] for agent, value in actions_and_logprob.items()}
#     log_probs = {agent: value[1] for agent, value in actions_and_logprob.items()}

#     print(actions)
    return actions, log_probs


# In[9]:


def evaluate(batch_obs, batch_acts, agent):
    """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
            batch_obs - the observations from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of observation)
            batch_acts - the actions from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of action)

        Return:
            V - the predicted values of batch_obs
            log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
    """
    # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
#     print(batch_obs)
    V = agent.critic(batch_obs).squeeze()

    # Calculate the log probabilities of batch actions using most recent actor network.
    # This segment of code is similar to that in get_action()
#     print(batch_obs.shape)
    output = agent.actor(batch_obs)
#     print(output.shape)
    m = Categorical(output)
#     print(type(m))
#     print(m)
    sample = m.sample()
#     print(type(dist))
#     print(dist)
#     print(dist.shape)
#     log_probs = dist.log_prob(m)
    log_probs = m.log_prob(sample)
#     print(log_probs.shape)

    # Return the value vector V of each observation in the batch
    # and log probabilities log_probs of each action in the batch
    return V, log_probs



# In[10]:


def _log_summary(key, agent, log_file):
    """
        Print to stdout what we've logged so far in the most recent batch.

        Parameters:
            None

        Return:
            None
    """
    # Calculate logging values. I use a few python shortcuts to calculate each value
    # without explaining since it's not too important to PPO; feel free to look it over,
    # and if you have any questions you can email me (look at bottom of README)
    delta_t = agent.logger['delta_t']
    agent.logger['delta_t'] = time.time_ns()
    delta_t = (agent.logger['delta_t'] - delta_t) / 1e9
    delta_t = str(round(delta_t, 2))

    t_so_far = agent.logger['t_so_far']
    i_so_far = agent.logger['i_so_far']
    avg_ep_lens = np.mean(agent.logger['batch_lens'])
    avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in agent.logger['batch_rews']])
    avg_actor_loss = np.mean([losses.float().mean() for losses in agent.logger['actor_losses']])

    # Round decimal places for more aesthetic logging messages
    avg_ep_lens = str(round(avg_ep_lens, 2))
    avg_ep_rews = str(round(avg_ep_rews, 2))
    avg_actor_loss = str(round(avg_actor_loss, 5))

    # Print logging statements
    print(flush=True)
    print(f"--------------------{key} Iteration #{i_so_far} --------------------", flush=True)
    print(f"Agent: {agent}", flush=True)
    print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
    print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
    print(f"Average Loss: {avg_actor_loss}", flush=True)
    print(f"Timesteps So Far: {t_so_far}", flush=True)
    print(f"Iteration took: {delta_t} secs", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(log_file)
    print(flush=True)

    with open(log_file, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([key,avg_ep_lens,avg_ep_rews,avg_actor_loss,t_so_far,delta_t])

    # Reset batch-specific logging data
    agent.logger['batch_lens'] = []
    agent.logger['batch_rews'] = []
    agent.logger['actor_losses'] = []


# In[11]:


# def run_training1(name_of_agent, max_eps, write_to_file=False):
def run_training1(hyperparameters):
    cyborg_version = CYBORG_VERSION
    sg = DroneSwarmScenarioGenerator(num_drones=num_drones, maximum_steps=500)
    cyborg = CybORG(sg, 'sim')
    env = PettingZooParallelWrapper(env=cyborg)
    observation = env.reset()
    
    current_date_time = datetime.now()
    formatted_date_time = current_date_time.strftime("%Y_%m_%d_%H_%M_%S")
    log_file = f"log_file_{formatted_date_time}.csv"
    with open(log_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Agent','Avg. Ep. Length','Avg. Ep. Return','Avg. Loss','Timesteps so far','Time taken'])
        
#     agents = {f"blue_agent_{agent}": PPO(env.observation_space(f"blue_agent_{agent}").shape[0]-(0*(num_drones-1)+0), len(env.get_action_space(f'blue_agent_{agent}')), 0.002, [0.9, 0.990], 0.99, 4, 0.2, False, None) for agent in range(num_drones)}
    start = datetime.now()
    
    models = {f"blue_agent_{agent}": PPO(policy_class_actor=FeedForwardNN_Actor, policy_class_critic=FeedForwardNN_Critic, env=env, **hyperparameters) for agent in range(num_drones)}
    t_so_far = 0 # Timesteps simulated so far
    i_so_far = 0 # Iterations ran so far
    total_timesteps = 20000000
    
    while t_so_far < total_timesteps:                                                                       # ALG STEP 2
        batch_obs_d, batch_acts_d, batch_log_probs_d, batch_rtgs_d, batch_lens_d = rollout(env=env, models=models, hyperparameters=hyperparameters)         # ALG STEP 3
#         print('after rollout')
#         print('batch_obs = ')
#         print(batch_obs_d)
#         print('batch_acts = ')
#         print(batch_acts_d)
#         print('batch_log_probs = ')
#         print(batch_log_probs_d)
#         print('batch_rtgs = ')
#         print(batch_rtgs_d)
#         print('batch_lens = ')
#         print(batch_lens_d)
        
#         print(len(batch_obs_d['blue_agent_0']), len(batch_acts_d['blue_agent_0']))
#         print(len(batch_obs_d['blue_agent_1']), len(batch_acts_d['blue_agent_1']))
#         print(len(batch_obs_d['blue_agent_2']), len(batch_acts_d['blue_agent_2']))
#         print(batch_obs_d['blue_agent_0'])
    
#         print("rollout complete")
    
        batch_lens = batch_lens_d 
        for key in batch_obs_d.keys():
            batch_obs = batch_obs_d[key]
            batch_acts = batch_acts_d[key]
            batch_log_probs = batch_log_probs_d[key]
            batch_rtgs = batch_rtgs_d[key]

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            models[key].logger['t_so_far'] = t_so_far
            models[key].logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
#             print(type(batch_obs), type(batch_acts))
#             print(batch_obs.shape, batch_acts.shape)
            V, _ = evaluate(batch_obs, batch_acts, models[key])
            
#             print('here')
#             print(type(V), V.shape)
#             print(type(batch_rtgs))
#             print(batch_rtgs)
#             print(batch_rtgs[key])
#             print(batch_rtgs)
            tensor_data = torch.stack(batch_rtgs)
#             print(type(tensor_data), tensor_data.shape)

            A_k = torch.stack(batch_rtgs) - V.detach()                                                                       # ALG STEP 5
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            
            for _ in range(hyperparameters['n_updates_per_iteration']):                                                       # ALG STEP 6 & 7
                V, curr_log_probs = evaluate(batch_obs, batch_acts, models[key])
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - hyperparameters['clip'], 1 + hyperparameters['clip']) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.CrossEntropyLoss()(V, torch.cat(batch_rtgs))
                models[key].actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                models[key].actor_optim.step()
                models[key].critic_optim.zero_grad()
                critic_loss.backward()
                models[key].critic_optim.step()
                models[key].logger['actor_losses'].append(actor_loss.detach())
    
            _log_summary(key, models[key], log_file)

            # Save our model if it's time
#             if i_so_far % models[key].save_freq == 0:
            if i_so_far % 10 == 0:
                torch.save(models[key].actor.state_dict(), './ppo_actor.pth')
                torch.save(models[key].critic.state_dict(), './ppo_critic.pth')

    end = datetime.now()
    print('REACHED HERE')
    print(end - start)
    
#     if technique == 'PPO':

# #         agents = {f"blue_agent_{agent}": PPO(env.observation_space(f"blue_agent_{agent}").shape[0], len(env.get_action_space(f'blue_agent_{agent}')), 0.002, [0.9, 0.990], 0.99, 4, 0.2, True, 'CybORG\Evaluation\submission\Models\\5110.pth') for agent in range(18)}
# #         agents = {f"blue_agent_{agent}": PPO(env.observation_space(f"blue_agent_{agent}").shape[0], len(env.get_action_space(f'blue_agent_{agent}')), 0.002, [0.9, 0.990], 0.99, 4, 0.2, True, None) for agent in range(num_drones)}
#         agents = {f"blue_agent_{agent}": PPO(env.observation_space(f"blue_agent_{agent}").shape[0]-(0*(num_drones-1)+0), len(env.get_action_space(f'blue_agent_{agent}')), 0.002, [0.9, 0.990], 0.99, 4, 0.2, False, None) for agent in range(num_drones)}
# #         agents = {f"blue_agent_{agent}": PPO(env.observation_space(f"blue_agent_{agent}").shape[0]-(3*(num_drones-1)+2), len(env.get_action_space(f'blue_agent_{agent}')), 0.002, [0.9, 0.990], 0.99, 4, 0.2, False, None) for agent in range(num_drones)}
#         if write_to_file:
#             file_name = str(inspect.getfile(CybORG))[:-7] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S")
#             print(f'Saving evaluation results to {file_name}_summary.txt and {file_name}_full.txt')
#         start = datetime.now()


        
        
        
        
        
        
#         total_reward = []
#         actions_log = []
#         obs_log = []
#         timestep = 0
#         update_timestep = 200
#         total_reward_n = 0
#         mean_total_reward = 0
#         for i in range(100000):
#             observation = env.reset()
# #             new_observation = remove_elements_from_dict_values(observation, num_drones)
#             new_observation = observation

#             r = []
#             a = []
#             o = []
#             for j in range(1000):
#                 timestep += 1
#                 actions = {agent_name: agent.get_action(new_observation[agent_name], agent.memory) for agent_name, agent in agents.items() if agent_name in env.agents}
#                 observation, rew, done, info = env.step(actions)
    
# #                 rew = {key: -value for key, value in rew.items()}
# #                 new_observation = remove_elements_from_dict_values(observation, num_drones)
#                 new_observation = observation

#                 for agent_name, agent in agents.items():
#                     if agent_name in actions:
#                         if agent_name in env.agents:
#                             agent.memory.rewards.append(rew[agent_name])
# #                             print("agent.memory.rewards = ", agent.memory.rewards)
#                             agent.memory.is_terminals.append(done[agent_name])
#                         else:
#                             agent.memory.rewards.append(-1)
#                             agent.memory.is_terminals.append(True)
#                 r.append(mean(rew.values()))
#                 if all(done.values()):
# #                     with open(file_name+'_1.txt', 'w') as data:
# #                         data.write(f'R:{r}\n')
#                     print("steps = ", j)
#                     break
#                 if write_to_file:
#                     a.append({agent_name: env.get_action_space(agent_name)[actions[agent_name]] for agent_name in actions.keys()})
#                     o.append({agent_name: observation[agent_name] for agent_name in observation.keys()})
                
#                 if timestep % update_timestep == 0:
#                     for agent_name, agent in agents.items():
#                         if agent_name in env.agents:
#                             agent.update()
#                             agent.memory.clear_memory()
#                     timestep = 0
# #             total_reward.append(sum(r))
#             total_reward_n += 1
#             if write_to_file:
#                 actions_log.append(a)
#                 obs_log.append(o)
#             mean_total_reward += (1/total_reward_n)*(sum(r)-mean_total_reward)   # for debugging mem overflow error
# #             mean_total_reward /= total_reward_n    # mem overflow error seen
#             print(i, ": ", mean_total_reward, sum(r))
#         ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i))
#         torch.save(agent.policy.state_dict(), ckpt)
#         print('Checkpoint saved')
#         end = datetime.now()
#         difference = end-start
  


# In[12]:


technique = 'PPO'
folder = 'Evaluation/submission_Jay'
ckpt_folder = os.path.join(os.getcwd(), folder, 'Models')
# print(ckpt_folder)
if not os.path.exists(ckpt_folder):
    os.makedirs(ckpt_folder)

run_training1(hyperparameters=hyperparameters)

# # Train or test, depending on the mode specified
# if args.mode == 'train':
#     run_training1(hyperparameters=hyperparameters)
# else:
#     run_training1(hyperparameters=hyperparameters)
# #     test(env=env, actor_model=args.actor_model)

# # run_training1(technique, 1000, True)


# In[ ]:


get_ipython().system('ls -l')


# In[ ]:




