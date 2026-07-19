import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_functional

# Parameters
input_size = 11 # state: board = 3 * 3 + action 1 * 2
num_of_neurons_in_layer1 = 128
num_of_neurons_in_layer2 = 64
output_size = 1 # Q(s,a)
gamma = 0.99 
MSELoss = nn.MSELoss()

class DQN (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if torch.cuda.is_available:
            self.device = torch.device('cpu') # 'cuda'
        else:
            self.device = torch.device('cpu')

        self.linear1 = nn.Linear(in_features=input_size, out_features=num_of_neurons_in_layer1)
        self.linear2 = nn.Linear(in_features=num_of_neurons_in_layer1, out_features=num_of_neurons_in_layer2)
        self.output = nn.Linear(in_features=num_of_neurons_in_layer2, out_features=output_size)
        
    # forward gets an array of n lines (n is number of legal moves) X 11.
    # 11 is the combination of the state (9 entries) and the legal actions (each action is a pair).
    # It returns n X 1 output array.
    def forward (self, x):
        # x is n X 11. n is the number of legal actions.
        # Each line consists of 9 inputs for the board state and a pair for each legal action.

        # Since self.linear1 is 11 X 128, the output is: n X 128
        linear_1_output = self.linear1(x)

        # Activation does not change dimensions, therefore linear_1_output_activated is n X 128
        linear_1_output_activated = torch_nn_functional.relu(linear_1_output)

        # Since self.linear2 is 128 X 64, the output is: n X 64
        linear_2_output = self.linear2(linear_1_output_activated)

        # Activation does not change dimensions, therefore linear_2_output_activated is n X 64
        linear_2_output_activated = torch_nn_functional.relu(linear_2_output)

        # Output layer is 64 X 1, therefore the output is n X 1.
        # This ouput is
        output = self.output(linear_2_output_activated)
        return output
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self):
        new_DQN = DQN()
        new_DQN.load_state_dict(self.state_dict())
        return new_DQN
    
    @staticmethod
    def loss (Q_value, rewards, Q_next_Values, Dones):
        Q_new = rewards + gamma * Q_next_Values * (1- Dones)
        return MSELoss(Q_value, Q_new)

    def __call__(self, states, actions):

        # self.forward receives a table.
        # Each row is a concatenation of the board state and a legal action.
        # board state is a 9 entries array with values of 1 for X, -1 for O and 0 for vacant.
        # possible actions are a pair (row, column) of a legal action.
        state_action = torch.cat((states,actions), dim=1)
        return self.forward(state_action)
