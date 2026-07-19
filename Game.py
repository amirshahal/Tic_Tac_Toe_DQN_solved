import argparse
from Graphics import *
from TicTacToe import TicTacToe
from State import State
from Human_Agent import Human_Agent
from Random_Agent import Random_Agent
from DqnAgent import DqnAgent

# Some paths used:
##PATH = 'Data\DQN_PARAM_1.pth'
# PATH = 'Data\DQN_PARAM_5_20K.pth'
## PATH = "Data\DQN_PARAM_Advanced_2.pth"

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player1", choices=["random", "human", "DQN"], required=True)
    parser.add_argument("--player2", choices=["random", "human", "DQN"], required=True)
    parser.add_argument("--path_dqn_player1")
    parser.add_argument("--path_dqn_player2")
    args_parsed = parser.parse_args()

    if args_parsed.player1 == "DQN" and args_parsed.path_dqn_player1 is None:
        args_parsed.path_dqn_player1 = "Data/DQN_PARAM_X.pth"

    if args_parsed.player2 == "DQN" and args_parsed.path_dqn_player2 is None:
        args_parsed.path_dqn_player2 = "Data/DQN_PARAM_O.pth"

    return args_parsed

def run_loop(player1, player2, env, graphics):
    pygame.init()
    clock = pygame.time.Clock()
    graphics.set_players(player1, player2)

    player = player1
    keep_running = True

    while keep_running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                keep_running = False

        # action is a pair(row, column)
        # if player is a DQN_Agent, this call gets the NN result for the current state.
        action = player(events=events, state=env.state, train=False)
        if action:
            env.move(action)
            player = switch_players(player, player1, player2)
            # pygame.time.wait(500)
            if env.state.end_of_game != 0:
                graphics(env.state)
                pygame.time.wait(2000)
                env.state.reset()
                player = player1
                # run = False

        graphics(env.state)
        clock.tick(FPS)


def switch_players(player, player1, player2):
    if player == player1:
        return player2
    else:
        return player1

def main ():
    env = TicTacToe(State())
    graphics = Graphics()
    args = read_args()

    if args.player1 == "random":
        player1 = Random_Agent(player=1, env=env, graphics=graphics)
    elif args.player1 == "human":
        player1 = Human_Agent(1, env=env, graphics=graphics)
    elif args.player1 == "DQN":
        player1 = DqnAgent(player=1,saved_nn_path=args.path_dqn_player1, train=False, env=env)
    else:
        raise f"Invalid player1 {args.player1}"

    if args.player2 == "random":
        player2 = Random_Agent(player=-1, env=env, graphics=graphics)
    elif args.player2 == "human":
        player2 = Human_Agent(-1, env=env, graphics=graphics)
    elif args.player2 == "DQN":
        player2 = DqnAgent(player=-1,saved_nn_path=args.path_dqn_player2, train=False, env=env)
    else:
        raise f"Invalid player2 {args.player2}"

    ## player2 = Random_Agent(player=-1, env=env, graphics=graphics)
    ## player2 = DqnAgent(player=1, saved_nn_path=PATH, train=False, env=env)
    ## player2 = Human_Agent(-1,env=env, graphics=graphics)
    graphics.set_players(player1, player2)
    run_loop(player1, player2, env, graphics)

if __name__ == '__main__':
    main()
