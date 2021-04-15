#!/usr/bin/python
# Anil Ozdemir
# The University of Sheffield
# 06.03.2021

# NOTES
# There are two reward types. It is experimental.
# TODO: documentation

import copy, itertools
import numpy as np
import gym
import matplotlib.pyplot as P
import seaborn as sns

import ipywidgets as widgets

class Robot():
    inWorld = lambda self, x: 0 <= x < self.Length
    move = np.array([(0,0), (-1,0), (0,-1), (1,0), (0,1)]) # Based on Matrix coords. NoMove, N, W, S, E,

    def __init__(self, position, robotID, Length):
        self.position = position
        self.ID       = robotID
        self.Length   = Length   # do we need length?

    def getSensor(self, W):
        # W: world matrix

        x,y = self.getPosition() # (row, cell)
        
        v    = [False] * 4
        v[0] = W[:x,:].any()
        v[1] = W[:,:y].any()
        v[2] = W[x+1:,:].any()
        v[3] = W[:,y+1:].any()
        c    = [False] * 4
        if x != 0:
            c[0] = W[x-1,y].any()
        if x != self.Length-1:
            c[2] = W[x+1,y].any()
        if y != 0:
            c[1] = W[x,y-1].any()
        if y != self.Length-1:
            c[3] = W[x,y+1].any()
        s = lambda v,c: 2 if c else (1 if v else 0)
        self.sensor = list(map(s,*(v,c)))
        return self.sensor

    def setPosition(self, action):
        is_move = False
        if action != 0: # if the action is not to stay
            newPos = self.position + self.move[action] # calc. possible new position
            if all(map(self.inWorld, newPos)): # if newPos in the world
                if not (self.sensor[action-1]) == 2: # if not colliding
                    self.position = newPos # assign the new position
                    is_move= True
        return is_move

    def getPosition(self):
        return tuple(self.position)

class gyMBlocks(gym.Env):
    
    STATES   = list(itertools.product([0,1,2], repeat = 4))
    getIndex = lambda self, x: self.STATES.index(tuple(x))

    def __init__(self, Length, nRobots, returnIndex = False, rewardType = 1, maxIter = 1000):

        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(4,), dtype=int)
        self.action_space      = gym.spaces.Discrete(5)
        self.Length            = Length
        self.nRobots           = nRobots
        self.returnIndex       = returnIndex
        self.rewardType        = rewardType
        self.maxIter           = maxIter
        self.reset(None)
                
    def initRobots(self, randomSeed=None):
        if randomSeed: np.random.seed(randomSeed)
            
        self.order    = np.random.permutation(self.nRobots)
        InitPos       = set(map(tuple,np.random.randint(self.Length, size=(self.nRobots,2))))
        self.robots   = []
        self.worldAll = []
        
        tryCount = 0
        while (len(InitPos) != self.nRobots):
            # can be more efficient, if to maintain empty spots, and select from empty spots
            InitPos_Re = set(map(tuple,np.random.randint(self.Length, size=(self.nRobots-len(InitPos),2))))
            InitPos    = InitPos.union(InitPos_Re)
            tryCount  += 1
            if tryCount > self.nRobots**2:  # try nRobots^2 times
                raise ValueError(f'>> Cannot initialise the environment. Tried {self.nRobots**2} times.')
                
        InitPosArray = np.array(list(map(list, InitPos)))[self.order] # initialised with order!
        self.world   = np.zeros((self.Length, self.Length), dtype=int)
        
        for RobotID, pos in enumerate(InitPosArray):
            self.world[tuple(pos)] = RobotID+1 # +1 is important, it is RobotID in {1,...,nRobots}
            self.robots.append(Robot(tuple(InitPosArray[RobotID]), RobotID+1, self.Length))    

        self.worldAll.append(copy.copy(self.world))

    def reset(self, randomSeed=None):
        self.initRobots(randomSeed) # initialise robots in self.world 
        self.iRobot             = 0
        self.iIter              = 0
        self.boundingBox       = []
        state = self.robots[self.iRobot].getSensor(self.world) # get the sensor readings from the first robot 
        if self.returnIndex:
            state = self.getIndex(state)
        return state

    def step(self, action):
        robot   = self.robots[self.iRobot]              # at this stage, we know the state & action is given
        old_pos = robot.getPosition()                   # keep old position in memory
        is_move = robot.setPosition(action)             # based on action, if possible (is_move=True), set new position 
        
        if is_move:                                     # if the robot moved
            self.world[old_pos]             = 0         # old coord turns 0 in World
            self.world[robot.getPosition()] = robot.ID  # new coord set in World
        
        assert (self.world == 0).sum() == self.Length**2-self.nRobots, ">> Overlap detected!" # check for overlap
        self.worldAll.append(copy.copy(self.world))                                           # make a copy to the list of all worlds
        
        self.iIter  += 1
        self.iRobot += 1        
        self.iRobot %= self.nRobots 

        done   = False
        reward = 0
        self.boundingBox.append(self.getBoundingBox(self.world))
        if self.boundingBox[-1][2] < self.boundingBox[-1][0]:
            done   = True
            if self.rewardType == 'exp-proportional' or self.rewardType == 1:
                reward = np.around(np.exp(-0.001 * (self.iIter/self.nRobots)), decimals=3) # exp-proportional reward
            if self.rewardType == 'proportional' or self.rewardType == 2:
                reward = (self.maxIter - self.iIter)/self.nRobots # proportional reward 

        state  = self.robots[self.iRobot].getSensor(self.world) # state for the next robot
        if self.returnIndex:
            state = self.getIndex(state)
        return state, reward, done, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            img = P.imshow(self.world)
            P.axis('off')
        elif mode == 'debug':
            print('iter: {} robot: {}'.format(self.iIter, self.iRobot))
        elif mode == 'array':
            return self.world
        else: # off
            pass
    
    def getBoundingBox(self, worldM=None):
        '''
        Pos is Pos for 1 Cycle is array of (NMODULE,2).
        Return L_1, L_2, N_Hole = L_1 * L_2  - NMODULE
        '''
        if type(worldM) == type(None):
            worldM = self.getWorldMatrix()
        AW = np.argwhere(worldM != 0)               # get non-zero elements (i.e. robots)
        L_x = np.max(AW[:,1]) - np.min(AW[:,1]) + 1
        L_y = np.max(AW[:,0]) - np.min(AW[:,0]) + 1 # x and y are opposite
        L_1 = np.min([L_x, L_y])
        L_2 = np.max([L_x, L_y])
        return np.array([L_1, L_2, L_1 * L_2  - self.nRobots])

    def getWorldMatrix(self):
        return self.world

    def getBoundingMatrix(self):
        return np.array(self.boundingBox)
    
    def debug(self):
        # can be calc. once, in reset()!
        cmap     = sns.cubehelix_palette(self.nRobots, dark=0.2, light=0.95, reverse=False)
        figsize  = np.clip((np.ceil(3*np.log(self.Length)),) * 2, 5, 12)
        fontsize = np.clip(np.ceil(20-3*np.log(self.Length)), 3, 15)
        
        fig, ax = P.subplots(figsize=figsize)
        sns.heatmap(self.world, annot=True, annot_kws={"fontsize":fontsize}, fmt="d",cmap=cmap, cbar=False, xticklabels=False, yticklabels=False);
        fig.tight_layout()
        
    def interactive(self):
        @widgets.interact(i=widgets.IntSlider(min=0, max=len(self.worldAll)-1, step=1, value=0))
        def f(i):
            # TODO can be calc. once, in reset()!
            # TODO mask instead of 0 in the world!
            cmap     = sns.cubehelix_palette(self.nRobots, start=.5, rot=-.8, dark=0.25, light=0.95, reverse=False)
            figsize  = np.clip((np.ceil(3*np.log(self.Length)),) * 2, 5, 12)
            fontsize = np.clip(np.ceil(20-3*np.log(self.Length)), 3, 15)
            
            fig, ax = P.subplots(figsize=figsize)
            sns.heatmap(self.worldAll[i], annot=True, annot_kws={"fontsize":fontsize}, fmt="d",cmap=cmap, cbar=False, xticklabels=False, yticklabels=False);
            fig.tight_layout()