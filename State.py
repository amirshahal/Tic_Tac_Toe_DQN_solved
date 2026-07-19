import numpy as np
import torch


class State:
    def __init__(self, board = None, player = 1):
        if board is not None:
            self.board = board
        else:
            self.board = self.init_board()
        self.player = player
        self.end_of_game = 0

    @staticmethod
    def init_board():
        board = np.zeros((3,3))
        return board
    
    def reset (self):
        self.board = self.init_board()
        self.player = 1
        self.end_of_game = 0

    def switch_players (self):
        if self.player == 1:
            self.player = -1
        else:
            self.player = 1

    def toTensor (self, device = torch.device('cpu')):
        array = self.board.reshape(-1)
        tensor = torch.tensor(array, dtype=torch.float32, device=device)
        return tensor
    
    @ staticmethod
    def tensorToState (state_tensor, player = 1):
        board = state_tensor.reshape([3,3]).cpu().numpy()
        return State(board, player)

    def __eq__(self, other) ->bool:
        is_equal = np.equal(self.board, other.board).all()
        return bool(is_equal)

    def __hash__(self) -> int:
        return hash(repr(self.board))
    
    def copy (self):
        newBoard = np.copy(self.board)
        return State(board=newBoard, player=self.player)
    
    def __str__(self) -> str:
        return str(self.board)
