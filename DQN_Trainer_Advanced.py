from DQN import DQN
from DqnAgent import DqnAgent
from Random_Agent import Random_Agent
from RandomAgentAdvanced import RandomAgentAdvanced
from TicTacToe import TicTacToe
from ReplayBuffer import ReplayBuffer
from State import State

import torch 

epochs = 500000
C = 300
batch = 64
learning_rate = 0.1
path = "Data\DQN_PARAM_Advanced.pth"
# path = "Data\DQN_PARAM_Advanced_2.pth"

# Todo: break main() into functions.
def main ():
    env = TicTacToe()
    player1 = DqnAgent(1, env=env)
    # player2 = Random_Agent(-1, env=env)
    player2 = RandomAgentAdvanced(player=-1, env=env)
    replay = ReplayBuffer()
    Q = player1.DQN
    Q_hat :DQN = Q.copy()
    Q_hat.train = False
    optim = torch.optim.SGD(Q.parameters(), lr=learning_rate)
    results = []
    result = [0,0,0] 
    loss = torch.tensor(0)
    for epoch in range(epochs):
        print (epoch, end="\r")
        state = State()
        while not env.is_end_of_game(state):
            action = player1.get_action(state, epoch=epoch)
            after_state, reward = env.next_state(state, action)
            if env.is_end_of_game(after_state):
                replay.push(state, action, reward, after_state, env.is_end_of_game(after_state))
                break
            after_action = player2.get_action(state=after_state)
            next_state, reward = env.next_state(after_state, after_action)
            replay.push(state, action, reward, next_state, env.is_end_of_game(next_state))
            state = next_state

            if epoch < 5000:
                continue
            states, actions, rewards, next_states, dones = replay.sample(batch)
            Q_values = Q(states, actions)
            next_actions = player1.get_actions(next_states, dones)
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states, next_actions)
            
            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
        if epoch % C == 0:
            Q_hat.load_state_dict(Q.state_dict())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.3f}, result: {result} ")
                  

        if epoch % 100 == 0:
            results.append(result)
            result = [0,0,0]
        else:
            if reward == 1:
                result[0] += 1
            elif reward < 0: 
                result[1] += 1
            else: 
                result[2] += 1

    
    player1.save_param(path)
    torch.save(results, f"Data/results_2.pth")
if __name__ == '__main__':
    main()
