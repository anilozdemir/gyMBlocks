#!/usr/bin/env python
# coding: utf-8

# Experiments for a Single Run (command-line input)

import sys, os, pickle, time
import pandas as pd
from joblib import Parallel, delayed
sys.path.append('../src')
from env import gyMBlocks
from agent import AGENT

## Variable to Test: `nHidden`

ALLHIDDEN = [16, 32, 64, 128, 256, 512]
nHidden   = ALLHIDDEN[int(sys.argv[1])] # command-line input

print(f'>> Running nHidden: {nHidden}')

## Run In Parallel (with Joblib)

envSize  = 5
nRobot   = 6
lR       = 0.001
nEpisode = 3000
gamma    = 0.99
maxIter  = 2000
rewType  = 1
def runAgent():
    exp = AGENT(envSize, nRobot, nHidden, lR, maxIter = maxIter, rewardType=rewType)
    exp.reinforce(nEpisode, gamma, returnDF=False);
    return exp

nExp = 10
st = time.time()
EXP  = Parallel(n_jobs=min(20,nExp),verbose=20)(delayed(runAgent)() for _ in range(nExp))
fn = time.time()
print(f'It took {fn-st:.3f}s')

# Save all data
filename = f'./data/nHidden-{nHidden}'

if not os.path.exists(filename): os.makedirs(filename)
    
for i in range(nExp):
    EXP[i].df.to_pickle(f"{filename}/df-{i}.pkl")
    with open(f'{filename}/agent-{i}', 'wb') as f:
        pickle.dump(EXP[i],f)    