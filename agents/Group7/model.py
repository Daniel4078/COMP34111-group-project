import random
from keras import layers, models
import tensorflow as tf
import numpy as np

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
    if game.get_board().has_ended():
        if game.get_board().get_winner() == agent_color:
            return 10  # Positive reward for winning
        else:
            return -10  # Negative reward for losing


def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        # Explore - choose a random action
        return np.random.randint(0, 36)
    else:
        # Exploit - choose the action with the highest Q-value
        Q_values = model.predict(state.reshape((1, 6, 6, 1)))
        return np.argmax(Q_values[0])


# Function to update Q-values using Q-learning update rule
def update_q_values(state, action, reward, done):
    target = reward
    if not done:
        next_state = States[move_num + 1]
        Q_values_next = model.predict(next_state.reshape((1, 6, 6, 1)))
        target += gamma * np.max(Q_values_next[0])
    Q_values = model.predict(state.reshape((1, 6, 6, 1)))
    Q_values[0, action] = target
    return Q_values


# Define the neural network architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(6, 6, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # Output for Q-values
    layers.Dense(36, activation='linear', name='q_values'),
    # layers.Dense(1, activation='sigmoid', name='winning_rate')  # Output for winning rate
])

# Compile the model with loss
model.compile(optimizer='adam',
              #   loss={'q_values': 'mean_squared_error', 'winning_rate': 'binary_crossentropy'},
              loss={'q_values': 'mean_squared_error'},
              loss_weights={'q_values': 1.0})
#   loss_weights={'q_values': 1.0, 'winning_rate': 0.5})

# # Define the neural network architecture
# model2 = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(6, 6, 1)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(36, activation='linear', name='q_values'),
# ])

# # Compile the model with loss
# model2.compile(optimizer='adam',
#               loss={'q_values': 'mean_squared_error'},
#               loss_weights={'q_values': 1.0})


# Training parameters
num_episodes = 30
win = 0

for episode in range(num_episodes):
    # Initialization
    game = Game(board_size=6)

    agent_color = random.choice([Colour.RED, Colour.BLUE])
    if agent_color == Colour.RED:
        player2 = Colour.BLUE
    else:
        player2 = Colour.RED
    
    start = True
    tiles = game.get_board().get_tiles()
    state = board_to_state(tiles)
    state = state.reshape((1, 6, 6, 1))
    total_reward = 0

    # To store every board state during one game
    States = []
    Actions = []

    while True:
        # Let Red starts first
        if agent_color == Colour.RED or start == False:
            action = choose_action(state, epsilon)
            row, col = divmod(action, 6)
            if state.reshape(6,6)[row, col] != 0:  # Check if the cell is already occupied
                continue  # Invalid move, skip to the next iteration
            
            # Make move
            game.get_board().set_tile_colour(row, col, agent_color)

            # Store the state after player1 move
            state = board_to_state(game.get_board().get_tiles())
            state = state.reshape((1, 6, 6, 1))

            # Add the action and state 
            States.append(state)
            Actions.append(action)

            if game.get_board().has_ended():
                break

        start = False
        # Player2 
        while True:
            action2 = np.random.randint(0, 36)
            row, col = divmod(action2, 6)
            if state.reshape(6,6)[row, col] == 0:
                game.get_board().set_tile_colour(row, col, player2)
                break

        # Store the state after player2 move
        state = board_to_state(game.get_board().get_tiles())
        state = state.reshape((1, 6, 6, 1))

        if game.get_board().has_ended():
            break

    # Give reward
    reward = calculate_reward(game, agent_color)
    for move_num in range(len(States) - 1, -1, -1):
        # Update Q-values using the Q-learning update rule
        Q_values = update_q_values(
            States[move_num], Actions[move_num], (0.9**(len(States) - move_num - 1)) * reward, (move_num + 1) == len(States))
        
        # Train the model on the updated Q-values
        model.train_on_batch(States[move_num].reshape((1, 6, 6, 1)), Q_values)

        total_reward += 0.9**(len(States) - move_num - 1) * reward

    print(Q_values)
    print(state.reshape(6,6))
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
# model.save('hex_agent_model.h5')
