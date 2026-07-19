import pygame
from State import State

FPS = 60

WIDTH, HEIGHT = 300, 400
ROWS, COLS = 3, 3
SQUARE_SIZE = 100
LINE_WIDTH = 2

H_WIDTH, H_HEIGHT = 300, 100
M_WIDTH, M_HEIGHT = 300, 300

#RGB
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
LIGHTGRAY = (211,211,211)
GREEN = (0, 128, 0)
CADETBLUE1 = (152,245,255)

class Graphics:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.header_surf = pygame.Surface((H_WIDTH, H_HEIGHT))
        self.main_surf = pygame.Surface((M_WIDTH, M_HEIGHT))
        self.player_x_type = None
        self.player_o_type = None
        pygame.display.set_caption('Tic Tac Toe')
        self.load_img()

    def draw (self, state: State):
        self.header_surf.fill(CADETBLUE1)
        self.main_surf.fill(LIGHTGRAY)
        
        self.draw_Lines()
        self.draw_pieces(state)

        self.write_player_types()

        if state.end_of_game == 1:
            self.write('Player X win', y=62, font_size=28)
        elif state.end_of_game == -1:
            self.write('Player O win', y=62, font_size=28)
        elif state.end_of_game == 2:
            self.write('Tie', y=62, font_size=28)
        else:
            if state.player == 1:
                self.write('Player X', y=62, font_size=28)
            else:
                self.write('Player O', y=62, font_size=28)

        self.screen.blit(self.header_surf, (0,0))
        self.screen.blit(self.main_surf, (0,100))
        pygame.display.update()

    def draw_Lines(self):
        for i in range(ROWS):
            pygame.draw.line(self.main_surf, BLACK, (i * SQUARE_SIZE, 0), 
                             (i * SQUARE_SIZE , M_HEIGHT), width=LINE_WIDTH)
            pygame.draw.line(self.main_surf, BLACK, (0, i * SQUARE_SIZE), 
                             (M_WIDTH, i * SQUARE_SIZE ), width=LINE_WIDTH)

    def set_players(self, player_x, player_o):
        self.player_x_type = self.agent_type(player_x)
        self.player_o_type = self.agent_type(player_o)

    def agent_type(self, player):
        return type(player).__name__.replace('_', ' ')

    def write_player_types(self):
        if self.player_x_type:
            self.write(f'Player X: {self.player_x_type}', y=6, font_size=22)
        if self.player_o_type:
            self.write(f'Player O: {self.player_o_type}', y=32, font_size=22)

    def write(self, txt, y=10, font_size=36):
        font = pygame.font.SysFont("Arial", font_size)
        txt_surf = font.render(txt, True, BLACK)
        self.header_surf.blit(txt_surf, (10,y))

    def draw_pieces(self, state: State):
        board = state.board
        for row in range(ROWS):
            for col in range(COLS):
                if board[row,col]!=0:
                    self.draw_piece((row,col), board[row,col])

    def draw_piece(self, row_col, player):
        if player == 1:
            img = self.x_img
        elif player == -1:
            img = self.o_img

        x, y = self.calc_pos(row_col)

        self.main_surf.blit(img,(x+10, y+10))

    def calc_row_col (self, pos):
        x, y = pos
        if y < 100:
            return None
        row = (y-H_HEIGHT) // SQUARE_SIZE
        col = x // SQUARE_SIZE
        return row, col

    def calc_pos (self, row_col):
        row, col = row_col
        x = col * SQUARE_SIZE
        y = row * SQUARE_SIZE
        return x, y

    def load_img (self):
        x_img = pygame.image.load("img/x_img.png")
        o_img = pygame.image.load("img/o_img.png") 
        self.x_img = pygame.transform.scale(x_img, (80, 80))
        self.o_img = pygame.transform.scale(o_img, (80, 80))

    def __call__(self, state):
        self.draw(state)
