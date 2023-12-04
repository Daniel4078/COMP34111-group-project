import random
from keras import layers, models
import tensorflow as tf
import numpy as np

import sys
sys.path.append(r"D:\COMP34111-group-project\src")

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


def calculate_reward(game, action, previous_state, agent_color):
    tiles = game.get_board().get_tiles()
    current_state = board_to_state(tiles)
    row, col = divmod(action, 6)

    if game.get_board().has_ended():
        if game.get_board().get_winner() == agent_color:  # Assuming the agent is RED
            return 10  # Positive reward for winning
        else:
            return -10  # Negative reward for losing
    elif previous_state[row, col] != 0:
        return -5  # Penalty for invalid moves
    return 0.001  # Progressive reward, which needs to be improved. 
              # We need to assign reward when it extend the longest chain.


def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        # Explore - choose a random action
        return np.random.randint(0, 36)
    else:
        # Exploit - choose the action with the highest Q-value
        Q_values = model.predict(state.reshape((1, 6, 6, 1)))
        return np.argmax(Q_values[0])

# Function to update Q-values using Q-learning update rule


def update_q_values(state, action, reward, next_state, done):
    target = reward
    if not done:
        Q_values_next = model.predict(next_state.reshape((1, 6, 6, 1)))
        target += gamma * np.max(Q_values_next[0])
    Q_values = model.predict(state.reshape((1, 6, 6, 1)))
    Q_values[0, action] = target
    return Q_values


# Define the neural network architecture with two outputs
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(6, 6, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # Output for Q-values
    layers.Dense(36, activation='linear', name='q_values'),
    # layers.Dense(1, activation='sigmoid', name='winning_rate')  # Output for winning rate
])

# Compile the model with multiple losses
model.compile(optimizer='adam',
              #   loss={'q_values': 'mean_squared_error', 'winning_rate': 'binary_crossentropy'},
              loss={'q_values': 'mean_squared_error'},
              loss_weights={'q_values': 1.0})
#   loss_weights={'q_values': 1.0, 'winning_rate': 0.5})

# Training parameters
num_episodes = 1

for episode in range(num_episodes):
    game = Game(board_size=11)
    agent_color = random.choice([Colour.RED, Colour.BLUE])
    if agent_color == Colour.RED:
        player2 = Colour.BLUE
    else:
        player2 = Colour.RED
    
    tiles = game.get_board().get_tiles()
    state = board_to_state(tiles)
    
    state = state.reshape((1, 11, 11, 1))
    total_reward = 0

    while True:
        previous_state = state.copy().reshape((11, 11))
        action = choose_action(state, epsilon)
        
        row, col = divmod(action, 11)
        if previous_state[row, col] != 0:  # Check if the cell is already occupied
            continue  # Invalid move, skip to the next iteration
        
        # Make move
        game.get_board().set_tile_colour(row, col, agent_color)

        # Store the state after player1 move
        next_state = board_to_state(game.get_board().get_tiles())
        next_state = next_state.reshape((1, 11, 11, 1))

        # Player2 
        while True:
            action2 = np.random.randint(0, 121)
            row, col = divmod(action2, 11)
            if state.reshape(11,11)[row, col] == 0:
                game.get_board().set_tile_colour(row, col, player2)
                break

        # Store the state after player2 move
        next_state2 = board_to_state(game.get_board().get_tiles())
        next_state2 = next_state2.reshape((1, 11, 11, 1))

        # Give reward
        reward = calculate_reward(game, action, previous_state, agent_color)
        
        # Update Q-values using the Q-learning update rule
        Q_values = update_q_values(
            state, action, reward, next_state, game.get_board().has_ended())

        # Train the model on the updated Q-values
        model.train_on_batch(state.reshape((1, 11, 11, 1)), Q_values)

        total_reward += reward
        state = next_state2

        
        
        if game.get_board().has_ended():
            print(state.reshape(11,11))
            break

    # Decay epsilon for exploration-exploitation trade-off
    epsilon *= epsilon_decay
    epsilon = max(min_epsilon, epsilon)

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Agent Colour: {agent_color}")
    print(f"Winner: {game.get_board().get_winner()}")

# Save the trained model for future use
# model.save('hex_agent_model.h5')
