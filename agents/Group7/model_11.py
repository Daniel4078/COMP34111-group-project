import random
from keras import layers, models
import tensorflow as tf
import numpy as np
import csv

import sys
sys.path.append(r"D:\Programming\COMP34111-group-project\src")

from Game import Game
from Colour import Colour


# Hyperparameters
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration-exploitation trade-off
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
    
    
def board_to_state(board_tiles):
    return np.array([[tile_to_state(tile) for tile in row] for row in board_tiles])


def calculate_reward(game, agent_color):
    if game.get_board().get_winner() == agent_color:
        return 10  # Positive reward for winning
    else:
        return -10  # Negative reward for losing


def choose_action(state, epsilon, model):
    if np.random.rand() < epsilon:
        # Explore - choose a random action
        while True:
            action =  np.random.randint(0, 121)
            row, col = divmod(action, 11)
            if state.reshape(11, 11)[row, col] == 0:
                return action, row, col

    num_selection = 1
    state_player = np.append(state, np.full((1, state.shape[1], state.shape[2], state.shape[3]), player2_num), axis=0)
    Q_values = model.predict(state_player.reshape((1, 2, 11, 11, 1)))
    indexes = np.argsort(Q_values[0])[::-1]
    while True:
        # Exploit - choose the action with the highest Q-value
        action = indexes[num_selection]

        # Check it has been occupied
        row, col = divmod(action, 11)
        if state.reshape(11, 11)[row, col] != 0:
            num_selection += 1
        else:
            return action, row, col

# Function to update Q-values using Q-learning update rule
def update_q_values(state, action, States, reward, done, model):
    target = reward
    if not done:
        next_state = States[move_num + 1]

        Q_values_next = model.predict(next_state.reshape((1, 2, 11, 11, 1)))
        target += gamma * np.max(Q_values_next[0])
    Q_values = model.predict(state.reshape((1, 2, 11, 11, 1)))
    Q_values[0, action] = target
    return Q_values


def create_model(input_shape=(2, 11, 11, 1)):
    model = models.Sequential()

    # First Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())

    # Second Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())

    # Third Convolutional Block
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(121, activation='linear', name='q_values')) # output layer

    return model


# # Define the neural network architecture
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(11, 11, 1)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(121, activation='linear', name='q_values'),
# ])

# Compile the model with loss
model = create_model()
model.compile(optimizer='adam',
              loss={'q_values': 'mean_squared_error'},
              loss_weights={'q_values': 1.0})

# # Define the second neural network architecture
# model2 = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(11, 11, 1)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(121, activation='linear', name='q_values'),
# ])

model2 = create_model()
# Compile the model with loss
model2.compile(optimizer='adam',
              loss={'q_values': 'mean_squared_error'},
              loss_weights={'q_values': 1.0})


# Training parameters
num_episodes = 10
win = 0
csv_file_path = 'board_evaluation.csv'

for episode in range(num_episodes):
    # Initialization
    game = Game(board_size=11)

    agent_color = random.choice([Colour.RED, Colour.BLUE])
    if agent_color == Colour.RED:
        player2 = Colour.BLUE
        player1_num = 1
        player2_num = 2
    else:
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

    while True:
        # Let Red starts first
        if agent_color == Colour.RED or start == False:
            # Add the state before move
            state_player = np.append(state, np.full((1, state.shape[1], state.shape[2], state.shape[3]), player1_num), axis=0)
            States.append(state_player)

            # Choose action
            action, row, col = choose_action(state, epsilon, model)

            # Make move
            game.get_board().set_tile_colour(row, col, agent_color)
            
            # Store the state_eval and action after player1 move
            state = board_to_state(game.get_board().get_tiles())
            state = state.reshape((1, 11, 11, 1))

            state_eval = np.append(state, np.full((1, state.shape[1], state.shape[2], state.shape[3]), player1_num), axis=0)
            States_eval.append(state_eval)
            Actions.append(action)

            if game.get_board().has_ended():
                break

        start = False
        # Player2 
        # Add the state before move
        state_player = np.append(state, np.full((1, state.shape[1], state.shape[2], state.shape[3]), player2_num), axis=0)
        States2.append(state_player)

        # Choose action
        action2, row, col = choose_action(state, epsilon, model2)
        # Make move
        game.get_board().set_tile_colour(row, col, player2)

        # Store the state_eval and action after player2 move
        state = board_to_state(game.get_board().get_tiles())
        state = state.reshape((1, 11, 11, 1))

        state2_eval = np.append(state, np.full((1, state.shape[1], state.shape[2], state.shape[3]), player2_num), axis=0)
        States2_eval.append(state2_eval)
        Actions2.append(action2)

        if game.get_board().has_ended():
            break

    # Give reward
    reward = calculate_reward(game, agent_color)
    for move_num in range(len(States) - 1, -1, -1):
        # Update Q-values using the Q-learning update rule
        Q_values = update_q_values(
            States[move_num], Actions[move_num], States, (0.9**(len(States) - move_num - 1)) * reward, (move_num + 1) == len(States), model)
        
        # Train the model on the updated Q-values
        model.train_on_batch(States[move_num].reshape((1, 2, 11, 11, 1)), Q_values)

        total_reward += 0.9**(len(States) - move_num - 1) * reward

    # Train the second model
    for move_num in range(len(States2) - 1, -1, -1):
        # Update Q-values using the Q-learning update rule
        Q_values2 = update_q_values(
            States2[move_num], Actions2[move_num], States2, (0.9**(len(States2) - move_num - 1)) * (-reward), (move_num + 1) == len(States2), model2)
        
        # Train the model on the updated Q-values
        model2.train_on_batch(States2[move_num].reshape((1, 2, 11, 11, 1)), Q_values2)

    # Train the step prediction model
    board_scores = []
    for move_num in range(len(States_eval)):
        if game.get_board().get_winner() == agent_color:
            board_scores.insert(0, 1 * (0.86**move_num))
        else:
            board_scores.insert(0, -1 * (0.86**move_num))

    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for move_num in range(len(States_eval)):
            # Save the state and board score to the CSV file
            csv_writer.writerow(list(States_eval[move_num]) + [board_scores[move_num]])

    board_scores = []
    for move_num in range(len(States2_eval)):
        if game.get_board().get_winner() != agent_color:
            board_scores.insert(0, 1 * (0.86**move_num))
        else:
            board_scores.insert(0, -1 * (0.86**move_num))

    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for move_num in range(len(States2_eval)):
            # Save the state and board score to the CSV file
            csv_writer.writerow(list(States2_eval[move_num]) + [board_scores[move_num]])


    print(Q_values)
    print(state.reshape(11, 11))
    # Decay epsilon for exploration-exploitation trade-off
    epsilon *= epsilon_decay
    epsilon = max(min_epsilon, epsilon)

    # Record the winning number
    if game.get_board().get_winner() == agent_color:
        win += 1
    
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Agent Colour: {agent_color}")
    print(f"Winner: {game.get_board().get_winner()}")

print(f"Winning rate: {win/(episode+1)}")

# Save the trained model for future use
model.save('hex_agent_model.keras')
model2.save('hex_agent_model2.keras')