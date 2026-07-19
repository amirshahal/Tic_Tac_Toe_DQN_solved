import math
import random
import torch
import torch.nn as nn
import numpy as np
from DQN import DQN
from State import State

# Epsilon Greedy parameters.
epsilon_start = 1
epsilon_final = 0.01
epsilon_decay = 5000

gamma = 0.99
MSELoss = nn.MSELoss()

# In order to play this constructor is called with train=False and parameters_path of a saved NN model.
class DqnAgent:
    def __init__(self, player = 1, saved_nn_path = None, train = True, env= None) -> None:
        self.DQN = DQN()
        if saved_nn_path:
            self.DQN.load_params(saved_nn_path)
        self.train(train)
        self.player = player
        self.env = env

    def train (self, train):
        # Set mode (train/not train) on the model and on all its layers.
        # In this case it does not matter b/c Linear and ReLU layers behave
        # the same in training and in evaluation. However, layers like:
        # nn.Dropout and nn.BatchNorm1d behave differently in training compare
        # to evaluation. Therefore, it is a good habit to set the network mode
        # to train / eval according its use.
        if train:
          self.DQN.train()
        else:
          self.DQN.eval()

    def get_action (self, state: State, epoch = 0, train = True):
        epsilon = self.epsilon_greedy(epoch)
        rnd = random.random()
        actions = self.env.legal_actions(state)
        if train and rnd < epsilon:
            return random.choice(actions)
        
        state_tensor = state.toTensor()
        action_np = np.array(actions)
        action_tensor = torch.from_numpy(action_np)

        # expand_state_action is an array (tensor) with the current board state (1 for X, -1 for O, 0 for vacant)
        # i.e. it has 9 columns. The same entry is repeated n times - one for each legal action.
        expand_state_tensor = state_tensor.unsqueeze(0).repeat((len(action_tensor),1))
        # state_action = torch.cat((expand_state_tensor, action_tensor ), dim=1)

        # Q_values holds the value for each legal action
        with torch.no_grad():
            Q_values = self.DQN(expand_state_tensor, action_tensor)

    # Take the action with the highest Q_value
        max_index = torch.argmax(Q_values)
        return actions[max_index]

    def get_actions (self, states, dones):
        actions = []
        for i, state in enumerate(states):
            if dones[i].item():
                # if the game is over, append a dummy action
                actions.append((0,0))
            else:
                actions.append(self.get_action(State.tensorToState(state), train=True)) #SARSA = True / Q-learning = False
        return torch.tensor(actions)

    # epsilon_greedy returns a value according epoch (all other parameters are constants).
    # final = 0.01
    # start = 1
    # Here are some function values:
    # epsilon_greedy(0) = 1
    # epsilon_greedy(100) = 0.99
    # epsilon_greedy(1000) = 0.821
    # epsilon_greedy(5000) = 0.374
    # epsilon_greedy(10000) = 0.144
    # epsilon_greedy(20000) = 0.028
    # epsilon_greedy(50000) = 0.01

    # During train (only!) we draw a random number from 0 to 1.
    # If this number is lower than epsilon_greedy(epoch) we try a random action.
    # Therefore, on the first epoch we will always try a random action (doing exploration).
    # As the number of epochs grows, we try less and less random actions; i.e. we do less exploration.

    @staticmethod
    def epsilon_greedy(epoch, start = epsilon_start, final=epsilon_final, decay=epsilon_decay):
        res = final + (start - final) * math.exp(-1 * epoch/decay)
        return res
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None, train=True, env=None):
        # Get the best action (the action with the highest value) using the NN.
        # i.e. we feed the state into the NN and get an action
        return self.get_action(state=state, train=train)


