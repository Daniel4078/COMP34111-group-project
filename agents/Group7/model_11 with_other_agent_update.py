import random
import keras
from keras import layers, models
import tensorflow as tf
import numpy as np
import csv
from EnemyAgent import EnemyAgent
import time
import SPOILER_new
from Game import Game
from Colour import Colour

# Load model
model = keras.models.load_model("hex_agent_model.keras")

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


def choose_action(state, epsilon, model):
    board = state.reshape(11, 11)
    if np.random.rand() < epsilon:
        # Explore - choose a random action
        while True:
            action = np.random.randint(0, 121)
            row, col = divmod(action, 11)
            if board[row][col] == 0:
                return action, row, col

    num_selection = 1
    Q_values = model.predict(state.reshape((1, 11, 11, 1)))[0]
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
def update_q_values(state, action, States, reward, done, model):
    target = reward
    if not done:
        next_state = States[move_num + 1]
        Q_values_next = model.predict(next_state.reshape((1, 11, 11, 1)))
        target += gamma * np.max(Q_values_next[0])
    Q_values = model.predict(state.reshape((1, 11, 11, 1)))
    Q_values[0, action] = target
    return Q_values


def update_q_values_illegal(state, action, reward, model):
    Q_values = model.predict(state.reshape((1, 11, 11, 1)))
    Q_values[0, action] = reward
    return Q_values


# Training parameters
num_episodes = 21*4
win = 0
total_training_time = 0
total_time = time.time()
csv_file_path = 'board_evaluation.csv'

for episode in range(num_episodes):
    # Initialization
    game = Game(board_size=11)
    agent_color = random.choice([Colour.RED, Colour.BLUE])
    # agent_color = Colour.RED
    # agent_color = Colour.BLUE

    if agent_color == Colour.RED:
        player2_color = "B"
        player2 = Colour.BLUE
        player1_num = 1
        player2_num = 2
    else:
        player2_color = "R"
        player2 = Colour.RED
        player1_num = 2
        player2_num = 1

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

    illegal_states = []
    illegal_moves = []

    # enemyAgent = EnemyAgent()
    enemyAgent = SPOILER_new.MCTSAgent()
    turn = 1

    # Set timeer
    startTime = time.time()
    while True:
        # Let Red starts first
        if agent_color == Colour.RED or start == False:
            # Add the state before move
            States.append(state)

            # Choose action
            action, row, col = choose_action(state, epsilon, model)

            # Make move
            game.get_board().set_tile_colour(row, col, agent_color)

            # Store the state_eval and action after player1 move
            state = board_to_state(game.get_board().get_tiles())
            state = state.reshape((1, 11, 11, 1))

            state_eval = np.append(state, np.full((1, state.shape[1], state.shape[2], state.shape[3]), player1_num),
                                   axis=0)
            States_eval.append(state_eval)
            Actions.append(action)

            if game.get_board().has_ended():
                break

        start = False
        # Player2 
        # Add the state before move
        States2.append(state)

        state_str = state_transfer(state.reshape(11, 11))

        # action2 = enemyAgent.run(player2_color, state_str, turn)
        action2 = enemyAgent.play_out(game.get_board(), player2_color)

        game.get_board().set_tile_colour(action2[0], action2[1], player2)

        turn += 1

        state = board_to_state(game.get_board().get_tiles())
        state = state.reshape((1, 11, 11, 1))

        state2_eval = np.append(state, np.full((1, state.shape[1], state.shape[2], state.shape[3]), player2_num),
                                axis=0)
        States2_eval.append(state2_eval)
        # Convert action
        action2 = action2[0] * 11 + action2[1]
        Actions2.append(action2)

        if game.get_board().has_ended():
            break
    run_time = time.time() - startTime


    startTime = time.time()
    # Give reward
    reward = calculate_reward(game, agent_color)
    for move_num in range(len(States) - 1, -1, -1):
        # Update Q-values using the Q-learning update rule
        Q_values = update_q_values(
            States[move_num], Actions[move_num], States, (0.9 ** (len(States) - move_num - 1)) * reward,
                                                         (move_num + 1) == len(States), model)

        # Train the model on the updated Q-values
        model.fit(States[move_num].reshape((1, 11, 11, 1)), Q_values, epochs=2)

    for move_num in range(len(States2) - 1, -1, -1):
        # Update Q-values using the Q-learning update rule
        Q_values = update_q_values(
            States2[move_num], Actions2[move_num], States2, (0.9 ** (len(States2) - move_num - 1)) * (-reward),
                                                            (move_num + 1) == len(States2), model)
        # Train the model on the updated Q-values
        model.fit(States2[move_num].reshape((1, 11, 11, 1)), Q_values, epochs=2)

    # Penalty for illegal moves
    for move_num in range(len(illegal_states)):
        # Update Q-values using the Q-learning update rule
        Q_values = update_q_values_illegal(
            illegal_states[move_num], illegal_moves[move_num], -1, model)
        # Train the model on the updated Q-values
        model.fit(illegal_states[move_num].reshape((1, 11, 11, 1)), Q_values, epochs=2)

    training_time = time.time() - startTime
    total_training_time += training_time

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

    # Decay epsilon for exploration-exploitation trade-off
    epsilon *= epsilon_decay
    epsilon = max(min_epsilon, epsilon)

    # Record the winning number
    if game.get_board().get_winner() == agent_color:
        win += 1

    print(f"Illegal moves in this round: {illegal_moves}")
    print(game.get_board().print_board())
    print(f"Episode: {episode + 1}, Total rounds: {turn}, Agent Colour: {agent_color}")
    print(f"Runing time: {run_time}, Training time: {training_time}")
    print(f"Winner: {game.get_board().get_winner()}")

print("")
print(f"Game statics (win/total): {win} / {episode + 1}")
print(f"Total training time: {total_training_time}")
print(f"Total time: {time.time()-total_time}")

# Save the trained model for future use
model.save('hex_agent_model.keras')
