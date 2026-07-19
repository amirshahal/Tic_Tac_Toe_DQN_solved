from collections import deque
import random
import torch
import numpy as np
from State import State

"""
ReplayBuffer is the agent’s memory of past moves, used to train the network on random batches of experience.

ReplayBuffer stores past game experiences so the DQN can train from them later.
Each stored item is:
  state: board before the move.
  action: move the DQN chose.
  reward: result after the move sequence.
  next_state: board after the move sequence.
  done: whether the game ended.

The replay buffer is used by DEN_Trainer.py.

The reason this helps:
1. It reuses old experience. 
Instead of learning only once from each move, the model can train many times from previous moves.

2. It breaks sequence correlation.
Consecutive moves in one game are highly related. Randomly sampling from memory gives the network a more mixed batch, which usually trains better.

3. It makes batch training possible.
Neural networks train better on batches like 64 examples at a time instead of one move at a time.

4. It stabilizes DQN learning.
DQN is already unstable because the target changes during training. Replay helps smooth that out.
"""

class ReplayBuffer:
    def __init__(self, capacity= 10000) -> None:
        # dequeue is doubled-ended queue. It is used here b/c it has fixed length.
        # After capacity is reached, each dequeue.append() removes the oldest (LIFO) entry
        # before adding a new entry.
        self.buffer = deque(maxlen=capacity)

    def push (self, state : State, action, reward, next_state : State, done):
        self.buffer.append((state.toTensor(), torch.from_numpy(np.array(action)), 
                            torch.tensor(reward), next_state.toTensor(), torch.tensor(done)))

    def push_tensors (self, state_tensor, action_tensor, reward_tensor, next_state_tensor, done):
        self.buffer.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done))
            
    def sample (self, batch_size):
        if batch_size > self.__len__():
            batch_size = self.__len__()
        state_tensors, action_tensor, reward_tensors, \
                   next_state_tensors, dones = zip(*random.sample(self.buffer, batch_size))
        states = torch.vstack(state_tensors)
        actions = torch.vstack(action_tensor)
        rewards = torch.vstack(reward_tensors)
        next_states = torch.vstack(next_state_tensors)
        done_tensor = torch.tensor(dones).long().reshape(-1,1)
        return states, actions, rewards, next_states, done_tensor

    def __len__(self):
        return len(self.buffer)

