from TicTacToe import TicTacToe
import pygame
import numpy as np
from Graphics import *
import random

class Random_Agent:
    def __init__(self, player, env: TicTacToe, graphics: Graphics = None):
        self.env = env
        self.player = player
        self.graphics = graphics

    def get_action(self, events=None, state = None, epoch=None, train = None):
        if state is None:
            state = self.env.state
        board = state.board
        indices = np.where(board == 0)
        actions = list(zip(indices[0], indices[1]))
        action = random.choice(actions)
        return action

    def get_state_action (self, state, epoch = None):
        action = self.get_action(state=state)
        next_state, reward = self.env.next_state(state, action)
        return action, reward, next_state


    def __call__(self, events=None, state=None, train=None):
        return self.get_action(events=events, state = state)
