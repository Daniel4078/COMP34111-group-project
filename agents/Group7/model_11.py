import random
from keras import layers, models
import tensorflow as tf
import numpy as np
import csv

import sys
sys.path.append(r"C:\Users\ttt\Desktop\COMP34111-group-project\src")

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
    Q_values = model.predict(state.reshape((1, 11, 11, 1)))
    while True:
        if num_selection == 1:
            # Exploit - choose the action with the highest Q-value
            action = np.argmax(Q_values[0])
        else:
            # Exploit - choose the action with the next highest Q-value
            action = np.argpartition(Q_values[0], -num_selection)[-num_selection]
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

        Q_values_next = model.predict(next_state.reshape((1, 11, 11, 1)))
        target += gamma * np.max(Q_values_next[0])
    Q_values = model.predict(state.reshape((1, 11, 11, 1)))
    Q_values[0, action] = target
    return Q_values


# Define the neural network architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(11, 11, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(121, activation='linear', name='q_values'),
])

# Compile the model with loss
model.compile(optimizer='adam',
              loss={'q_values': 'mean_squared_error'},
              loss_weights={'q_values': 1.0})

# Define the second neural network architecture
model2 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(11, 11, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(121, activation='linear', name='q_values'),
])

# Compile the model with loss
model2.compile(optimizer='adam',
              loss={'q_values': 'mean_squared_error'},
              loss_weights={'q_values': 1.0})


# Training parameters
num_episodes = 200
win = 0
csv_file_path = 'board_evaluation.csv'

for episode in range(num_episodes):
    # Initialization
    game = Game(board_size=11)

    agent_color = random.choice([Colour.RED, Colour.BLUE])
    if agent_color == Colour.RED:
        player2 = Colour.BLUE
    else:
        player2 = Colour.RED
    
    start = True
    tiles = game.get_board().get_tiles()
    state = board_to_state(tiles)
    state = state.reshape((1, 11, 11, 1))

    total_reward = 0
    
    # To store every board state during one game
    States = []
    Actions = []

    States2 = []
    Actions2 = []

    while True:
        # Let Red starts first
        if agent_color == Colour.RED or start == False:
            # Choose action
            action, row, col = choose_action(state, epsilon, model)
            # Make move
            game.get_board().set_tile_colour(row, col, agent_color)
            
            # Store the state after player1 move
            state = board_to_state(game.get_board().get_tiles())
            state = state.reshape((1, 11, 11, 1))

            # Add the action and state 
            States.append(state)
            Actions.append(action)

            if game.get_board().has_ended():
                break

        start = False
        # Player2 
        action2, row, col = choose_action(state, epsilon, model2)
        # Make move
        game.get_board().set_tile_colour(row, col, player2)

        # Store the state after player2 move
        state = board_to_state(game.get_board().get_tiles())
        state = state.reshape((1, 11, 11, 1))

        # Add the action and state 
        States2.append(state)
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
        model.train_on_batch(States[move_num].reshape((1, 11, 11, 1)), Q_values)

        total_reward += 0.9**(len(States) - move_num - 1) * reward

    # Train the second model
    for move_num in range(len(States2) - 1, -1, -1):
        # Update Q-values using the Q-learning update rule
        Q_values2 = update_q_values(
            States2[move_num], Actions2[move_num], States2, (0.9**(len(States2) - move_num - 1)) * (-reward), (move_num + 1) == len(States2), model2)
        
        # Train the model on the updated Q-values
        model2.train_on_batch(States2[move_num].reshape((1, 11, 11, 1)), Q_values2)

    # Train the step prediction model
    board_scores = []
    for move_num in range(len(States)):
        if game.get_board().get_winner() == agent_color:
            board_scores.insert(0, 1 * (0.86**move_num))
        else:
            board_scores.insert(0, -1 * (0.86**move_num))
    # model3.fit(np.array(States).reshape(-1, 6, 6, 1), np.array(board_scores), epochs=5)
    with open(csv_file_path, 'a') as csv_file:
        csv_writer = csv.writer(csv_file)
        for move_num in range(len(States)):
            # Save the state and board score to the CSV file
            csv_writer.writerow(list(States[move_num].reshape(-1, 11, 11, 1)) + [board_scores[move_num]])

    board_scores = []
    for move_num in range(len(States2)):
        if game.get_board().get_winner() != agent_color:
            board_scores.insert(0, 1 * (0.86**move_num))
        else:
            board_scores.insert(0, -1 * (0.86**move_num))
    # model3.fit(np.array(States2).reshape(-1, 6, 6, 1), np.array(board_scores), epochs=5)
    with open(csv_file_path, 'a') as csv_file:
        csv_writer = csv.writer(csv_file)
        for move_num in range(len(States2)):
            # Save the state and board score to the CSV file
            csv_writer.writerow(list(States2[move_num].reshape(-1, 11, 11, 1)) + [board_scores[move_num]])


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