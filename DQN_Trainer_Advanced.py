import argparse
import shutil
from DQN import DQN
from DqnAgent import DqnAgent
from Random_Agent import Random_Agent
from RandomAgentAdvanced import RandomAgentAdvanced
from TicTacToe import TicTacToe
from ReplayBuffer import ReplayBuffer
from State import State

from datetime import timedelta
from pathlib import Path
import time
import torch

EPOCHS = 500000
C = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.01
LEARNING_RATE_TOKEN = str(LEARNING_RATE).replace(".", "p")
RESULTS_WINDOW = 100
CHECKPOINT_WINDOW = 100000


def format_elapsed_time(seconds):
    elapsed = timedelta(seconds=seconds)
    hours, remainder = divmod(elapsed.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = elapsed.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def backup_existing_file(path):
    source_path = Path(path)
    if not source_path.exists():
        return None

    backup_index = 0
    while True:
        backup_path = Path(f"{source_path}.{backup_index:03}")
        if not backup_path.exists():
            print(f"Saving existing {source_path} as {backup_path}")
            shutil.copy2(source_path, backup_path)
            return backup_path
        backup_index += 1


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player_to_train", choices=["X", "O"], required=True)
    parser.add_argument(
        "--opponent_player",
        choices=["Random_Agent", "RandomAgentAdvanced"],
        default="RandomAgentAdvanced",
    )
    parser.add_argument("--optimizer", choices=["SGD", "ADAM"], default="SGD")
    args_parsed = parser.parse_args()
    return args_parsed


def build_output_paths(train_player, optimizer_name):
    player_label = "X" if train_player == 1 else "O"
    common_suffix = (
        f"epochs_{EPOCHS}.optimizer_{optimizer_name}."
        f"lr_{LEARNING_RATE_TOKEN}.pth"
    )
    model_path = f"Data/DQN_PARAM_Advanced_{player_label}.{common_suffix}"
    results_path = (
        f"Data/results_advanced_{player_label.lower()}.{common_suffix}"
    )
    return model_path, results_path


def create_optimizer(q, optimizer_name):
    if optimizer_name == "SGD":
        return torch.optim.SGD(q.parameters(), lr=LEARNING_RATE)
    if optimizer_name == "ADAM":
        return torch.optim.Adam(q.parameters(), lr=LEARNING_RATE)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def checkpoint_path(path, epoch):
    source_path = Path(path)
    checkpoint_label = f"{(epoch + 1) // 1000}K"
    return source_path.with_name(
        f"{source_path.stem}.{checkpoint_label}{source_path.suffix}"
    )


def save_checkpoint_outputs(dqn_player, path, results, results_path, epoch):
    model_checkpoint_path = checkpoint_path(path, epoch)
    results_checkpoint_path = checkpoint_path(results_path, epoch)
    backup_existing_file(model_checkpoint_path)
    dqn_player.save_param(model_checkpoint_path)
    backup_existing_file(results_checkpoint_path)
    torch.save(results, results_checkpoint_path)
    print(f"Saved checkpoint parameters to {model_checkpoint_path}")
    print(f"Saved checkpoint results to {results_checkpoint_path}")


def train_from_replay(replay, q, q_hat, dqn_player, optim):
    states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)
    q_values = q(states, actions)
    next_actions = dqn_player.get_actions(next_states, dones)
    with torch.no_grad():
        q_hat_values = q_hat(next_states, next_actions)

    loss = q.loss(q_values, rewards, q_hat_values, dones)
    loss.backward()
    optim.step()
    optim.zero_grad()
    return loss


def play_training_game(env, dqn_player, opponent_player, replay, train_player,
                       epoch, q, q_hat, optim):
    state = State()
    loss = None

    while not env.is_end_of_game(state):
        if state.player != train_player:
            opponent_action = opponent_player.get_action(state=state)
            state, _ = env.next_state(state, opponent_action)
            if env.is_end_of_game(state):
                break

        dqn_state = state
        dqn_action = dqn_player.get_action(dqn_state, epoch=epoch)
        after_dqn_state, reward = env.next_state(dqn_state, dqn_action)
        dqn_reward = reward * train_player
        if env.is_end_of_game(after_dqn_state):
            replay.push(
                dqn_state, dqn_action, dqn_reward, after_dqn_state, True
            )
            return after_dqn_state, loss

        opponent_action = opponent_player.get_action(state=after_dqn_state)
        next_state, reward = env.next_state(after_dqn_state, opponent_action)
        dqn_reward = reward * train_player
        done = env.is_end_of_game(next_state)
        replay.push(dqn_state, dqn_action, dqn_reward, next_state, done)
        state = next_state

        if len(replay) >= BATCH_SIZE:
            loss = train_from_replay(replay, q, q_hat, dqn_player, optim)

    return state, loss


def update_result(result, state):
    if state.end_of_game == 1:
        result[0] += 1
    elif state.end_of_game == -1:
        result[1] += 1
    else:
        result[2] += 1


def print_progress(epoch, loss, result, start_time):
    elapsed = format_elapsed_time(time.perf_counter() - start_time)
    print(
        f"{elapsed} Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.3f}, "
        f"result [X wins, O wins, Tie]: {result} "
    )


def train(env, dqn_player, opponent_player, replay, q, q_hat, optim,
          train_player, start_time, path, results_path):
    results = []
    result = [0, 0, 0]
    loss = torch.tensor(0)

    for epoch in range(EPOCHS):
        state, current_loss = play_training_game(
            env, dqn_player, opponent_player, replay, train_player,
            epoch, q, q_hat, optim
        )
        if current_loss is not None:
            loss = current_loss

        if epoch % C == 0:
            q_hat.load_state_dict(q.state_dict())

        update_result(result, state)

        if (epoch + 1) % RESULTS_WINDOW == 0:
            print_progress(epoch, loss, result, start_time)
            results.append(result)
            result = [0, 0, 0]

        if (epoch + 1) % CHECKPOINT_WINDOW == 0:
            save_checkpoint_outputs(
                dqn_player, path, results, results_path, epoch
            )

    return results


def save_training_outputs(dqn_player, path, results, results_path):
    backup_existing_file(path)
    dqn_player.save_param(path)
    backup_existing_file(results_path)
    torch.save(results, results_path)
    print(f"Training complete. Saved parameters to {path}")
    print(f"Saved training results to {results_path}")


def main():
    start_time = time.perf_counter()
    env = TicTacToe()
    args = read_args()

    if args.player_to_train == "X":
        train_player = 1
    elif args.player_to_train == "O":
        train_player = -1
    else:
        raise ValueError(
            f"Do not know how to train {args.player_to_train}, "
            "please provide X or O"
        )

    path, results_path = build_output_paths(train_player, args.optimizer)
    dqn_player = DqnAgent(train_player, env=env)

    if args.opponent_player == "RandomAgentAdvanced":
        opponent_player = RandomAgentAdvanced(_player=-train_player, env=env)
    elif args.opponent_player == "Random_Agent":
        opponent_player = Random_Agent(-train_player, env=env)
    else:
        raise ValueError(
            "DQN_Trainer_Advanced::main() does not know how to handle "
            f"args.opponent_player {args.opponent_player}"
        )

    replay = ReplayBuffer()
    q = dqn_player.DQN
    q_hat: DQN = q.copy()
    q_hat.eval()
    optim = create_optimizer(q, args.optimizer)
    player_label = "X" if train_player == 1 else "O"
    print(
        "DQN_Trainer_Advanced(): "
        f"Training DQN as player {player_label} for {EPOCHS} epochs. "
        f"Optimizer: {args.optimizer}. "
        f"Saving to {path}"
    )

    results = train(
        env, dqn_player, opponent_player, replay, q, q_hat, optim,
        train_player, start_time, path, results_path
    )

    save_training_outputs(dqn_player, path, results, results_path)


if __name__ == '__main__':
    main()
