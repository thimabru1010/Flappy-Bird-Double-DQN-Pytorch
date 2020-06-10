EPSILON_INIT = 1
EPSILON_FINAL = 0.001
#EPSILON_DECAY = 3000000
EPSILON_DECAY = 0.99

import copy
import math
import random
import torch

from dqn2 import DQN

class Agent_Clipped(object):
    def __init__(self, cuda=False):
        self.epsilon = EPSILON_INIT
        self.net1 = DQN(n_action=2)
        self.target1 = DQN(n_action=2)
        self.net2 = DQN(n_action=2)
        self.target2 = DQN(n_action=2)
        if cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.net1.to(device)
            self.target1.to(device)
            self.net2.to(device)
            self.target2.to(device)
        self.update_targets()

    def Q1(self, input, action=None, target=False, argmax=False):
        #print('entrou Q1')
        f = self.target1 if target else self.net1
        if action:
            ret = f(input)[0][action].item()
        else:
            if argmax:
                ret = f(input).argmax(dim=1)[0].item()
            else:
                ret = f(input).max(dim=1)[0].item()
        return ret

    def Q2(self, input, action=None, target=False, argmax=False):
        #print('entrou Q2')
        f = self.target2 if target else self.net2
        if action:
            ret = f(input)[0][action].item()
        else:
            if argmax:
                ret = f(input).argmax(dim=1)[0].item()
            else:
                ret = f(input).max(dim=1)[0].item()
        return ret

    def make_action1(self, state, explore=False):
        if explore and self.epsilon > random.random():
            # make it more likely do nothing
            action = random.choice([0, 1])
        else:
            # Talvez seja bom tirar isso
            #Q = self.net1(state)
            action = self.Q1(state, argmax=True)
        return action

    def make_action2(self, state, explore=False):
        if explore and self.epsilon > random.random():
            # make it more likely do nothing
            action = random.choice([0, 1])
        else:
            # Talvez seja bom tirar isso
            #Q = self.net2(state)
            action = self.Q2(state, argmax=True)
        return action

    def update_targets(self):
        state_dict = copy.deepcopy(self.net1.state_dict())
        self.target1.load_state_dict(state_dict)

        state_dict = copy.deepcopy(self.net2.state_dict())
        self.target2.load_state_dict(state_dict)

    def update_epsilon(self):
        if self.epsilon > EPSILON_FINAL:
            #self.epsilon -= (EPSILON_INIT - EPSILON_FINAL) / EPSILON_DECAY
            self.epsilon *= EPSILON_DECAY

    @property
    def parameters1(self):
        return self.net1.parameters()

    @property
    def parameters2(self):
        return self.net2.parameters()
