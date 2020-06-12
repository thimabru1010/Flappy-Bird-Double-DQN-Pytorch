TARGET_UPDATE_CYCLE = 50
LOGGING_CYCLE = 1
BATCH = 32
# Gamma
DISCOUNT = 0.99
LEARNING_RATE = 1e-4
OBSERV = 50000
CAPACITY = 50000
SAVE_MODEL_CYCLE = 5000
# TEST
# OBSERV = 500
# CAPACITY = 500
# SAVE_MODEL_CYCLE = 50


import time
import os
import random
from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt
import numpy as np

class Trainer_Clipped(object):
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.seed = random.randint(0, 20180818)
        self.optimizer1 = optim.Adam(agent.parameters1, lr=LEARNING_RATE)
        self.optimizer2 = optim.Adam(agent.parameters2, lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(capacity=CAPACITY)
        self.total_step = 0

    def run(self, device='cpu', buffer=False, explore=False):
        """Run an episode and buffer"""
        self.env.reset()
        self.env.env.seed(self.seed)
        state = self.env.get_screen()
        states = np.asarray([state for _ in range(4)]) # shape (4, 84, 84)
        step = 0
        accumulated_reward = 0
        while True:
            action = self.agent.make_action1(torch.Tensor([states]).to(device), explore=explore)
            state_next, reward, done = self.env.step(action)
            states_next = np.concatenate([states[1:, :, :], [state_next]], axis=0)
            step += 1
            accumulated_reward += reward
            if buffer:
                self.buffer.append(states, action, reward, states_next, done)
            states = states_next
            if explore == False:
                # Render the screen to see training
                self.env.env.render()
            if done:
                break
        return accumulated_reward, step

    def _fill_buffer(self, num, device='cpu'):
        start = time.time()
        while self.buffer.size < num:
            self.run(device, buffer=True, explore=True)
            print('Fill buffer: {}/{}'.format(self.buffer.size, self.buffer.capacity))
        print('Filling buffer takes {:.3f} seconds'.format(time.time() - start))

    def train(self, device='cpu'):
        # Sp
        self.env.change_record_every_episode(100000000)
        self._fill_buffer(OBSERV, device)
        if self.env.record_every_episode:
            self.env.change_record_every_episode(self.env.record_every_episode)

        episode = 0
        total_accumulated_rewards = []
        while 'training' != 'converge':
        #while episode <= 500:
            self.env.reset()
            state = self.env.get_screen()
            states = np.asarray([state for _ in range(4)]) # shape (4, 84, 84)
            step_prev = self.total_step
            accumulated_reward = 0
            done = False
            n_flap = 0
            n_none = 0
            while not done:
                #### --------------------
                #### Add a new transition
                # Calculates actions based on e-greedy
                action = self.agent.make_action1(torch.Tensor([states]).to(device), explore=True)

                state_next, reward, done = self.env.step(action)

                states_next = np.concatenate([states[1:, :, :], [state_next]], axis=0)
                self.total_step += 1
                accumulated_reward += reward
                self.buffer.append(states, action, reward, states_next, done)
                states = states_next
                #### --------------------

                #### --------------------
                #### Training step
                start = time.time()
                # prepare training data
                minibatch = self.buffer.sample(n_sample=BATCH)
                _states = [b[0] for b in minibatch]
                _actions = [b[1] for b in minibatch]
                _rewards = [b[2] for b in minibatch]
                _states_next = [b[3] for b in minibatch]
                _dones = [b[4] for b in minibatch]

                ys = []
                for i in range(len(minibatch)):
                    terminal = _dones[i]
                    r = _rewards[i]
                    if terminal:
                        y = r
                    else:
                        # Double DQN
                        # Nessa parte usamos a e_greedy para tomar a ação, ou sempre nos baseamos no argmax da propria rede?
                        s_t_next = torch.Tensor([_states_next[i]]).to(device)
                        # Calculates de action with self.net1
                        online_act1 = self.agent.make_action1(s_t_next)
                        # Calculates de max value for online_act1
                        max_value1 = self.agent.Q1(s_t_next, online_act1, target=True)
                        # Calculates de action with self.net2
                        # online_act2 = self.agent.make_action2(s_t_next)
                        # Calculates de max value for online_act2
                        max_value2 = self.agent.Q2(s_t_next, online_act1, target=True)
                        # Index 0 network1, Index 1 network 2
                        max_values = [max_value1,max_value2]
                        index = np.argmin(np.asarray(max_values))
                        # Calculates the total reward with the target network using the action calculated form the other network (self.net)
                        # Both network1 and network 2 shares the same target
                        y = r + DISCOUNT * max_values[index]
                    ys.append(y)
                ys = torch.Tensor(ys).to(device)

                # Render the screen to see training
                #self.env.env.render()

                # Apply gradient on network 1
                #print('Traning network 1...')
                self.optimizer1.zero_grad()
                input = torch.Tensor(_states).to(device)

                output1 = self.agent.net1(input) # shape (BATCH, 2)

                actions_one_hot = np.zeros([BATCH, 2])
                actions_one_hot[np.arange(BATCH), _actions] = 1.0
                actions_one_hot = torch.Tensor(actions_one_hot).to(device)
                ys_hat = (output1 * actions_one_hot).sum(dim=1)
                loss1 = F.smooth_l1_loss(ys_hat, ys)
                loss1.backward()
                self.optimizer1.step()

                # Apply gradient on network 2
                #print('Traning network 2...')
                self.optimizer2.zero_grad()
                input = torch.Tensor(_states).to(device)

                output2 = self.agent.net2(input) # shape (BATCH, 2)

                actions_one_hot = np.zeros([BATCH, 2])
                actions_one_hot[np.arange(BATCH), _actions] = 1.0
                actions_one_hot = torch.Tensor(actions_one_hot).to(device)
                ys_hat = (output2 * actions_one_hot).sum(dim=1)
                loss2 = F.smooth_l1_loss(ys_hat, ys)
                loss2.backward()
                self.optimizer2.step()
                #### --------------------

                # logging
                if action == 0:
                    n_flap += 1
                else:
                    n_none += 1

                if done and self.total_step % LOGGING_CYCLE == 0:
                    log = '[{}, {}] alive: {}, reward: {}, F/N: {}/{}, loss1: {:.4f}, loss2: {:.4f}, epsilon: {:.4f}, time: {:.3f}, network: Q{}'.format(
                        episode,
                        self.total_step,
                        self.total_step - step_prev,
                        accumulated_reward,
                        n_flap,
                        n_none,
                        loss1.item(),
                        loss2.item(),
                        self.agent.epsilon,
                        time.time() - start,
                        index+1)
                    print(log)

                self.agent.update_epsilon()
                if self.total_step % TARGET_UPDATE_CYCLE == 0:
                    #print('[Update target network]')
                    self.agent.update_targets()

                if self.total_step % SAVE_MODEL_CYCLE == 0:
                    print('[Save model]')
                    self.save(id=self.total_step)
                    if len(total_accumulated_rewards) > 0:
                        self.save_graph_rewards(episode, total_accumulated_rewards)

            # Keep the accumulated_reward for all the episodes
            total_accumulated_rewards.append(accumulated_reward)
            episode += 1

    def save_graph_rewards(self, episodes, total_accumulated_rewards):
        #fig = plt.figure()
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.xlabel('Episodes')
        plt.ylabel('Total reward')
        episodes_x = np.linspace(0, episodes, episodes)
        ax.plot(episodes_x, np.ones(episodes)*0, color='red', label='ref')
        ax.plot(episodes_x, total_accumulated_rewards, color='turquoise', label='real')
        ax.legend(loc='lower left')
        if not os.path.exists('tmp/graphs'):
            os.makedirs('tmp/graphs')
        plt.savefig(f'tmp/graphs/Total_rewards_ep={episodes}.png')


    def save(self, id):
        filename = 'tmp/models1/model_{}.pth.tar'.format(id)
        dirpath = os.path.dirname(filename)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        checkpoint = {
            'net': self.agent.net1.state_dict(),
            'target': self.agent.target1.state_dict(),
            'optimizer': self.optimizer1.state_dict(),
            'total_step': self.total_step
        }
        torch.save(checkpoint, filename)

        # Only saves Q1
        filename = 'tmp/models2/model_{}.pth.tar'.format(id)
        dirpath = os.path.dirname(filename)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        checkpoint = {
            'net': self.agent.net2.state_dict(),
            'target': self.agent.target2.state_dict(),
            'optimizer': self.optimizer2.state_dict(),
            'total_step': self.total_step
        }
        torch.save(checkpoint, filename)

    def load(self, filename, device='cpu'):
        # FAZER AJUSTES DEPOIS PARA EVALUATE
        # LEMBRAR DO USO DE DUAS REDES, LOGO DOIS OPTIMIZERS
        ckpt = torch.load(filename, map_location=lambda storage, loc: storage)
        ## Deal with the missing of bn.num_batches_tracked
        net_new = OrderedDict()
        tar_new = OrderedDict()

        for k, v in ckpt['net'].items():
            for _k, _v in self.agent.net1.state_dict().items():
                if k == _k:
                    net_new[k] = v

        for k, v in ckpt['target'].items():
            for _k, _v in self.agent.target1.state_dict().items():
                if k == _k:
                    tar_new[k] = v

        self.agent.net1.load_state_dict(net_new)
        self.agent.target1.load_state_dict(tar_new)
        ## -----------------------------------------------

        self.optimizer1.load_state_dict(ckpt['optimizer'])
        self.total_step = ckpt['total_step']
