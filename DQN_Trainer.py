import argparse
from DQN import DQN
from DqnAgent import DqnAgent
from Random_Agent import Random_Agent
from RandomAgentAdvanced import RandomAgentAdvanced
from TicTacToe import TicTacToe
from ReplayBuffer import ReplayBuffer
from State import State

from datetime import timedelta
import time
import torch 

epochs = 150000
C = 300
batch_size = 64
learning_rate = 0.001
results_window = 100
path_x = "Data\\DQN_PARAM_X.pth"
path_o = "Data\\DQN_PARAM_O.pth"

def format_elapsed_time(seconds):
    elapsed = timedelta(seconds=seconds)
    hours, remainder = divmod(elapsed.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = elapsed.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

# Todo: break main() to functions. Current main() length is too long! (Amir, 20260719)
def main ():
    args = read_args()

    if args.player_to_train == "X":
        train_player = 1
    elif args.player_to_train == "O":
        train_player = -1
    else:
        raise f"Do not know how to train {args.player_to_train}, please provide X or O"

    start_time = time.perf_counter()
    path = path_x if train_player == 1 else path_o
    env = TicTacToe()
    dqn_player = DqnAgent(train_player, env=env)
    if args.opponent_player == "RandomAgentAdvanced":
        opponent_player = RandomAgentAdvanced(_player=-train_player, env=env)
    elif args.opponent_player == "Random_Agent":
        opponent_player = Random_Agent(-1, env=env)
    else:
        raise f"DQN_Trainer::main() do not know hot to handle args.opponent_player {args.opponent_player}"


    replay = ReplayBuffer()

    # Q is the NN that is being trained
    Q = dqn_player.DQN

    # Q_hat is the NN that is used to evaluate Q
    Q_hat :DQN = Q.copy()
    Q_hat.eval()

    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    loss = None
    updates = 0
    wins = 0
    losses = 0
    ties = 0
    player_label = "X" if train_player == 1 else "O"
    print(f"Training DQN as player {player_label} for {epochs} epochs. Saving to {path}")
       
    for epoch in range(epochs):
        state = State()
        reward = 0
        while not env.is_end_of_game(state):
            if state.player != train_player:
                opponent_action = opponent_player.get_action(state=state)
                state, reward = env.next_state(state, opponent_action)

                # reward is set according the player.
                # Positive reward for X is negative reward for O
                # and vice versa.
                reward *= train_player
                if env.is_end_of_game(state):
                    break

            dqn_state = state
            dqn_action = dqn_player.get_action(dqn_state, epoch=epoch)
            after_dqn_state, reward = env.next_state(dqn_state, dqn_action)
            reward *= train_player
            if env.is_end_of_game(after_dqn_state):
                replay.push(dqn_state, dqn_action, reward, after_dqn_state, True)
                break

            opponent_action = opponent_player.get_action(state=after_dqn_state)

            next_state, reward = env.next_state(after_dqn_state, opponent_action)
            reward *= train_player
            replay.push(dqn_state, dqn_action, reward, next_state, env.is_end_of_game(next_state))
            state = next_state

            # start train the NN only after we have at least batch_size lines in the replay buffer
            if len(replay) < batch_size:
                continue

            # The following operations are performed on batch_size samples.
            # 1. Get a batch_size samples.
            states, actions, rewards, next_states, dones = replay.sample(batch_size)

            # 2. For each tuple among the batch_size tuples, get the Q_values (batch_size # of values)
            #   which is calculated using the Q NN (forward).
            Q_values = Q(states, actions)

            # 3. get batch_size next_actions.
            next_actions = dqn_player.get_actions(next_states, dones)
            with torch.no_grad():
                # 4. Calculate batch_size of Q_hat_Values
                Q_hat_Values = Q_hat(next_states, next_actions)
            
            # loss is one value, MSE of the two vectors: Q_values and Q_new.
            #  Q_new := rewards + gamma * Q_next_Values * (1- dones). Q_next_Values are calculated using the best
            #  action (the action that maximize the reward). We want to bring Q_values as closer as possible to
            #  the Q_next_Values.
            # i.e. we want to minimize the differences between the Q_values which
            # we got from the replay_buffer and the Q_hat_values.
            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()

            # improve W, b according gradients.
            # This is done for Q only (and not for Q-hat).
            optim.step()

            # get ready for next optimization round.
            optim.zero_grad()
            updates += 1

        if epoch % C == 0:
            # Update Q_hat to be similar to Q (gain what we learned so far).
            Q_hat.load_state_dict(Q.state_dict())

        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            ties += 1

        if (epoch + 1) % results_window == 0:
            if loss is None:
                loss_text = "not started"
            else:
                loss_text = f"{loss.item():.4f}"
            elapsed = format_elapsed_time(time.perf_counter() - start_time)
            print(f"{elapsed} Epoch {epoch + 1}/{epochs}, replay={len(replay)}, updates={updates}, loss={loss_text}, last {results_window}: wins={wins}, losses={losses}, ties={ties}")
            wins = 0
            losses = 0
            ties = 0

    dqn_player.save_param(path)
    print(f"Training complete. Saved parameters to {path}")

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player_to_train", choices=["X", "O"], required=True)
    parser.add_argument("--opponent_player", choices=["Random_Agent", "RandomAgentAdvanced"],
                        default="RandomAgentAdvanced")
    args_parsed = parser.parse_args()
    return args_parsed

if __name__ == '__main__':
    main()
