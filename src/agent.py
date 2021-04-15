#!/usr/bin/python
# Anil Ozdemir
# The University of Sheffield
# 06.03.2021

# NOTES
# AGENT class uses some parts of https://github.com/PacktPublishing/PyTorch-1.x-Reinforcement-Learning-Cookbook/blob/master/Chapter08/chapter8/reinforce.py for REINFORCE implementation
# TODO: documentation

from env import gyMBlocks
import pickle, time
import torch
import torch.nn as nn
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as P
import pandas as pd
import numpy as np

# A simple function for calculating rolling sum
def rolling_sum(a, n=10, normalise=False):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    if normalise:
        return ret[n - 1:]/n
    else:
        return ret[n - 1:]
    
# REINFORCE Agent class     
# The network policy is a built-in one hidden-layer MLP
class AGENT():
    def __init__(self, envSize = 5, nRobot = 6 , n_hidden = 50, lr = 0.001, maxIter = 1000, rewardType = 1, randSeed = 0):
       
        self.env = gyMBlocks(envSize, nRobot, returnIndex = True, rewardType=rewardType, maxIter = maxIter)
        self.env.seed(randSeed)
        nStates  = len(self.env.STATES)
        nAction  = self.env.action_space.n
        
        self.model = nn.Sequential(
                        nn.Linear(nStates, n_hidden),
                        nn.ReLU(),
                        nn.Linear(n_hidden, nAction),
                        nn.Softmax(dim=0),
                )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.OH        = torch.eye((nStates))
        
        # self attribute
        self.envSize    = envSize
        self.nRobot     = nRobot
        self.n_hidden   = n_hidden
        self.lr         = lr
        self.maxIter    = maxIter
        self.rewardType = rewardType
        self.randSeed   = randSeed 
        
    def predict(self, state):
        return self.model(torch.Tensor(state))


    def update(self, advantages, log_probs):
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, advantages):
            policy_gradient.append(-log_prob * Gt)

        loss = torch.stack(policy_gradient).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, s):
        probs = self.predict(s)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob

    def reinforce(self, nEpisode, gamma=0.99, returnDF=False, progressBar=False):
        total_reward_episode = [0] * nEpisode
        logs = []
        env = self.env
        for episode in trange(nEpisode,disable=not progressBar):
            log_probs = []
            rewards = []
            state = env.reset()
    
            while True:
                action, log_prob = self.get_action(self.OH[state])
                next_state, reward, is_done, _ = env.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)
    
                if is_done or env.iIter >= env.maxIter:
                    total_reward_episode[episode] += reward
                    Gt = 0
                    pw = 0
    
                    returns = []
                    for t in range(len(rewards)-1, -1, -1):
                        Gt += gamma ** pw * rewards[t]
                        pw += 1
                        returns.append(Gt)
    
                    returns = returns[::-1]
                    returns = torch.tensor(returns)
                    self.update(returns, log_probs)
    
                    break
                state = next_state
            logs.append([episode, is_done, reward, env.iIter, env.boundingBox[-1]])   
        self.df = pd.DataFrame(logs, columns=['ep', 'done', 'reward', 'epLength', 'bbox'])
        if returnDF:
            return self.df
        
    def play(self, envSize = 15, nRobots = 20):
        with torch.no_grad():
            maxIter = envSize * nRobots * 20
            env = gyMBlocks(envSize, nRobots, returnIndex = True, maxIter= maxIter)
            env.seed(0)
            state = env.reset()
            while True:
                action, log_prob = self.get_action(self.OH[state])
                next_state, reward, is_done, _ = env.step(action)
                if is_done or env.iIter >= env.maxIter:
                    break
                state = next_state
            env.interactive()
            
    def plot(self, nBin = 50, normalise = True):
        P.plot(rolling_sum(np.array(self.df.reward), n = nBin, normalise=normalise));
        if normalise:
            P.ylim([-0.05,1.05])
    
    def saveDF(self,filename):
        self.df.to_pickle(filename)
        
    def getTabularPolicy(self,plot=True):
        TABLE = []
        with torch.no_grad():
            for i, row in enumerate(self.OH):
                action, log_prob = self.get_action(row)
                TABLE.append([i, action])
        if plot == True:
            P.hist(np.array(TABLE)[:,1])
            P.xticks(np.arange(0,5), ['stop', 'north', 'west', 'south', 'east']);
            P.xlabel('actions')
            P.ylabel('count')
        return np.array(TABLE)   
    
    
