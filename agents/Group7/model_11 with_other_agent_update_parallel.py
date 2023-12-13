import random
import keras
import numpy as np
import csv
# from EnemyAgent import EnemyAgent
import time

import tensorflow as tf
import SPOILER_new
from Game import Game
from Colour import Colour
from dask.distributed import Client, LocalCluster, progress
import dask

# Hyperparameters
gamma = 0.9  # Discount factor
epsilon = 0.45  # Exploration-exploitation trade-off
epsilon_decay = 0.995
min_epsilon = 0.01


# Assume the Hex board is represented as a 2D array, where 0 represents an empty cell,
# 1 represents Player 1, and 2 represents Player 2.

# Function to choose an action using epsilon-greedy policy
def tile_to_state(tile):
    colour = tile.get_colour()
    if colour == Colour.RED:
        return 1  # Assuming RED represents Player 1
    elif colour == Colour.BLUE:
        return 2  # Assuming BLUE represents Player 2
    else:
        return 0  # Assuming None or another value represents an empty tile


def state_transfer(state):
    transformed_state = []
    for row in state:
        transformed_row = []
        for tile in row:
            if tile == 1:
                transformed_row.append("R")
            elif tile == 2:
                transformed_row.append("B")
            else:
                transformed_row.append("0")
        transformed_state.append(transformed_row)
    return np.array(transformed_state)


def board_to_state(board_tiles):
    return np.array([[tile_to_state(tile) for tile in row] for row in board_tiles])


def calculate_reward(game, agent_color):
    if game.get_board().get_winner() == agent_color:
        return 10  # Positive reward for winning
    else:
        return -10  # Negative reward for losing


def choose_action(state, epsilon, model, illegal_states, illegal_moves):
    board = state.reshape(11, 11)
    if np.random.rand() < epsilon:
        # Explore - choose a random action
        while True:
            action = np.random.randint(0, 121)
            row, col = divmod(action, 11)
            if board[row][col] == 0:
                return action, row, col

    num_selection = 1
    Q_values = model.predict(state.reshape((1, 11, 11, 1)), verbose=0)[0]
    indexes = np.argsort(Q_values)[::-1]
    while True:
        # Exploit - choose the action with the highest Q-value
        action = indexes[num_selection]
        # Check it has been occupied
        row, col = divmod(action, 11)
        if board[row][col] != 0:
            # Store the illegal moves
            illegal_states.append(state)
            illegal_moves.append(action)
            num_selection += 1
        else:
            return action, row, col


# Function to update Q-values using Q-learning update rule
def update_q_values(state, action, States, reward, done, model, move_num):
    target = reward
    if not done:
        next_state = States[move_num + 1]
        Q_values_next = model.predict(next_state.reshape((1, 11, 11, 1)), verbose=0)
        target += gamma * np.max(Q_values_next[0])
    Q_values = model.predict(state.reshape((1, 11, 11, 1)), verbose=0)
    Q_values[0, action] = target
    return Q_values


def update_q_values_illegal(state, action, reward, model):
    Q_values = model.predict(state.reshape((1, 11, 11, 1)), verbose=0)
    Q_values[0, action] = reward
    return Q_values


def swap_color(x):
    result = []
    for i in range(11):
        temp = []
        for j in range(11):
            if x[i, j] == 1:
                temp.append(2)
            elif x[i, j] == 2:
                temp.append(1)
            else:
                temp.append(0)
        result.append(temp)
    return result


def mirror_board(states, actions):
    states_T = []
    actions_T = []
    for move_num in range(len(states)):
        action = actions[move_num]
        state = states[move_num]
        row, col = divmod(action, 11)
        action_T = col * 11 + row
        temp = state.transpose()
        state_T = np.array(swap_color(temp))
        states_T.append(state_T)
        actions_T.append(action_T)
    return states_T, actions_T


def generate_Q(States, Actions, reward, model, states_total, Q_total):
    for move_num in range(len(States) - 1, -1, -1):
        # Update Q-values using the Q-learning update rule
        Q_values = update_q_values(
            States[move_num], Actions[move_num], States, (0.9 ** (len(States) - move_num - 1)) * reward,
                                                         (move_num + 1) == len(States), model, move_num)
        states_total.append(States[move_num])
        Q_total.append(Q_values)
    return states_total, Q_total

def generate_Q_il(illegal_states, illegal_moves, model, states_total, Q_total):
    # Penalty for illegal moves
    for move_num in range(len(illegal_states)):
        # Update Q-values using the Q-learning update rule
        Q_values = update_q_values_illegal(
            illegal_states[move_num], illegal_moves[move_num], -1, model)
        states_total.append(illegal_states[move_num])
        Q_total.append(Q_values)
    return states_total, Q_total

def write_csv(csv_file_path, States_eval, States2_eval, game, agent_color):
    # Train the step prediction model
    board_scores = []
    for move_num in range(len(States_eval)):
        if game.get_board().get_winner() == agent_color:
            board_scores.insert(0, 1 * (0.86 ** move_num))
        else:
            board_scores.insert(0, -1 * (0.86 ** move_num))

    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for move_num in range(len(States_eval)):
            # Save the state and board score to the CSV file
            csv_writer.writerow(list(States_eval[move_num]) + [board_scores[move_num]])

    board_scores = []
    for move_num in range(len(States2_eval)):
        if game.get_board().get_winner() == agent_color:
            board_scores.insert(0, -1 * (0.86 ** move_num))
        else:
            board_scores.insert(0, 1 * (0.86 ** move_num))

    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for move_num in range(len(States2_eval)):
            # Save the state and board score to the CSV file
            csv_writer.writerow(list(States2_eval[move_num]) + [board_scores[move_num]])


def play_game():
    model = keras.models.load_model("hex_agent_model.keras")
    # Initialization
    game = Game(board_size=11)
    agent_color = random.choice([Colour.RED, Colour.BLUE])
    # agent_color = Colour.RED
    # agent_color = Colour.BLUE

    if agent_color == Colour.RED:
        player2_color = "B"
        player2 = Colour.BLUE
    else:
        player2_color = "R"
        player2 = Colour.RED

    start = True
    tiles = game.get_board().get_tiles()
    state = board_to_state(tiles)
    state = state.reshape((1, 11, 11, 1))

    total_reward = 0

    # To store every board state during one game
    States = []
    States_eval = []
    Actions = []

    States2 = []
    States2_eval = []
    Actions2 = []

    states_total = []
    Q_total = []
    illegal_states = []
    illegal_moves = []

    # enemyAgent = EnemyAgent()
    enemyAgent = SPOILER_new.MCTSAgent()
    turn = 0

    # Set timeer
    startTime = time.time()
    while True:
        # Let Red starts first
        if agent_color == Colour.RED or start == False:
            # Add the state before move
            States.append(state)

            # Choose action
            action, row, col = choose_action(state, epsilon, model, illegal_states, illegal_moves)

            # Make move
            game.get_board().set_tile_colour(row, col, agent_color)

            # Store the state_eval and action after player1 move
            state = board_to_state(game.get_board().get_tiles())
            state = state.reshape((1, 11, 11, 1))

            States_eval.append(state)
            Actions.append(action)

            if game.get_board().has_ended():
                break

        start = False
        # Player2 
        # Add the state before move
        States2.append(state)

        # action2 = enemyAgent.run(player2_color, state_str, turn)
        action2 = enemyAgent.play_out(game.get_board(), player2_color)

        game.get_board().set_tile_colour(action2[0], action2[1], player2)

        turn += 1

        state = board_to_state(game.get_board().get_tiles())
        state = state.reshape((1, 11, 11, 1))

        States2_eval.append(state)
        # Convert action
        action2 = action2[0] * 11 + action2[1]
        Actions2.append(action2)

        if game.get_board().has_ended():
            break

    # Give reward
    reward = calculate_reward(game, agent_color)
    States_T, Actions_T = mirror_board(States, Actions)
    il_state_T, il_action_T = mirror_board(illegal_states, illegal_moves)
    States2_T, Actions2_T = mirror_board(States2, Actions2)
    states_total, Q_total = generate_Q(States, Actions, reward, model, states_total, Q_total)
    states_total, Q_total = generate_Q(States2, Actions2, -reward, model, states_total, Q_total)
    states_total, Q_total = generate_Q_il(illegal_states, illegal_moves, model, states_total, Q_total)
    states_total, Q_total = generate_Q(States_T, Actions_T, reward, model, states_total, Q_total)
    states_total, Q_total = generate_Q(States2_T, Actions2_T, -reward, model, states_total, Q_total)
    states_total, Q_total = generate_Q_il(il_state_T, il_action_T, model, states_total, Q_total)
    # write_csv(csv_file_path, States_eval, States2_eval, game, agent_color)
    return states_total, Q_total


def main(cluster):
    # Training parameters
    global epsilon
    model = keras.models.load_model("hex_agent_model.keras")
    num_episodes = 5
    total_training_time = 0
    total_time = time.time()
    csv_file_path = 'board_evaluation.csv'
    client = Client(cluster)
    futures = []
    for episode in range(num_episodes):
        for _ in range(8):
            future = client.submit(play_game)
            futures.append(future)
        print(progress(futures))
        results = client.gather(futures)
        print("Running time:", time.time() - total_time)
        States, Q_values = zip(*results)
        States = np.array(States)
        Q_values = np.array(Q_values)

        # Decay epsilon for exploration-exploitation trade-off
        epsilon *= epsilon_decay
        epsilon = max(min_epsilon, epsilon)

        training_time = time.time()
        for _ in range(8):
            for i in range(len(States)):
                states_reshaped = States[i].reshape(len(States[i]), 11, 11, 1)
                q_values_reshaped = Q_values[i].reshape(len(Q_values[i]), 121)
                model.train_on_batch(states_reshaped, q_values_reshaped)
        total_training_time += time.time() - training_time
    client.close()
    print("")
    print(f"Total training time: {total_training_time}")
    print(f"Total time: {time.time() - total_time}")
    # Save the trained model for future use
    model.save('hex_agent_model.keras')


if __name__ == '__main__':
    cluster = LocalCluster(n_workers=8)
    main(cluster)
