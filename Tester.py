import argparse
from TicTacToe import TicTacToe
from State import State
from Random_Agent import Random_Agent
from RandomAgentAdvanced import RandomAgentAdvanced
from DqnAgent import DqnAgent

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player_to_train", choices=["X", "O",], required=True)
    parser.add_argument("--player2type", choices=["RandomAgent", "RandomAgentAdvanced"],
                        default="RandomAgentAdvanced")
    parser.add_argument("--path_dqn_to_train")
    parser.add_argument("--number_of_games", type=int, default=1000)
    args_parsed = parser.parse_args()

    if args_parsed.path_dqn_to_train is None:
        if args_parsed.player_to_train == "X":
            args_parsed.path_to_train = "Data/DQN_PARAM_X.pth"
        elif args_parsed.player_to_train == "O":
            args_parsed.path_to_train = "Data/DQN_PARAM_O.pth"
        else:
            raise f"read_args(): Do not know how to train {args_parsed.player_to_train}"

    return args_parsed

def main ():
    env = TicTacToe(State())

    args = read_args()

    if args.player_to_train == "X":
        player1 = DqnAgent(1, env=env, saved_nn_path=args.path_to_train, train=False)
        player2type = -1
    elif args.player_to_train == "O":
        player1 = DqnAgent(-1, env=env, saved_nn_path=args.path_to_train, train=False)
        player2type = 1
    else:
        raise f"Do not know how to train {args.player_to_train}, please provide X or O"

    # PATH = "Data\DQN_PARAM_5_20K.pth"
    if args.player2type == "RandomAgent":
        player2 = Random_Agent(player2type, env, graphics=None)
    elif args.player2type == "RandomAgentAdvanced":
        player2 = RandomAgentAdvanced(player2type, env, graphics=None)
    else:
        raise f"main(): do not know how to train {args.player2type}"

    x_win_count = 0
    o_win_count = 0
    tie_count = 0
        
    for n in range(args.number_of_games):
        state = State()
        player = player1
        while not env.is_end_of_game(state):
            action = player.get_action(state=state, train=False)
            state, _ = env.next_state(state,action)
            player = switch_players(player, player1, player2)
        if state.end_of_game == 1:
            x_win_count +=1
        elif state.end_of_game == -1:
            o_win_count += 1
        else:
            tie_count +=1
        state.reset()
    print(f"Testing {args.player_to_train} player, played {args.number_of_games} games. X won {x_win_count} times,"
        f"O won {o_win_count} times, {tie_count} games were tied.")

def switch_players(player, player1, player2):
    if player == player1:
        return player2
    else:
        return player1

if __name__ == '__main__':
    main()
