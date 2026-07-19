from TicTacToe import TicTacToe
from Graphics import *

class Human_Agent:
    def __init__(self, player, env: TicTacToe, graphics: Graphics):
        self.env = env
        self.player = player
        self.graphics = graphics

    def get_action(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                action = self.graphics.calc_row_col(pos)
                if self.env.legal(self.env.state, action):
                    return action
        return None

    def __call__(self, events=None, state=None, env=None, train=None):
        return self.get_action(events)
    

