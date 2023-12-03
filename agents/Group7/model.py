import numpy as np
import tensorflow as tf
from keras import layers, models

# Hyperparameters
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration-exploitation trade-off
epsilon_decay = 0.995
min_epsilon = 0.01

# Assume the Hex board is represented as a 2D array, where 0 represents an empty cell,
# 1 represents Player 1, and 2 represents Player 2.

# Function to choose an action using epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        # Explore - choose a random action
        return np.random.randint(0, 121)
    else:
        # Exploit - choose the action with the highest Q-value
        Q_values = model.predict(state.reshape((1, 11, 11, 1)))
        return np.argmax(Q_values[0])

# Function to update Q-values using Q-learning update rule
def update_q_values(state, action, reward, next_state, done):
    target = reward
    if not done:
        Q_values_next = model.predict(next_state.reshape((1, 11, 11, 1)))
        target += gamma * np.max(Q_values_next[0])
    Q_values = model.predict(state.reshape((1, 11, 11, 1)))
    print(Q_values)
    Q_values[0, action] = target
    return Q_values

# Define the neural network architecture with two outputs
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(11, 11, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(121, activation='linear', name='q_values'),  # Output for Q-values
    # layers.Dense(1, activation='sigmoid', name='winning_rate')  # Output for winning rate
])

# Compile the model with multiple losses
model.compile(optimizer='adam',
            #   loss={'q_values': 'mean_squared_error', 'winning_rate': 'binary_crossentropy'},
              loss={'q_values': 'mean_squared_error'},
              loss_weights={'q_values': 1.0})
            #   loss_weights={'q_values': 1.0, 'winning_rate': 0.5})

# Training parameters
num_episodes = 10

for episode in range(num_episodes):
    state = np.random.randint(0, 3, size=(11, 11))
    state = state.reshape((1, 11, 11, 1))
    total_reward = 0

    for step in range(121):  # Maximum number of steps in a Hex game
        action = choose_action(state, epsilon)
        next_state = state.copy()
        row, col = divmod(action, 11)
        if next_state[0, row, col, 0] == 0:  # Check if the chosen cell is empty
            next_state[0, row, col, 0] = 1  # Assume Player 1 makes a move
        else:
            continue  # Invalid move, choose a different action

        # Simulate the environment and get the reward
        # In a real scenario, you would interact with the actual Hex game environment
        reward = 1  # Placeholder reward, you need to define the reward based on game rules

        # Update Q-values using the Q-learning update rule
        Q_values = update_q_values(state, action, reward, next_state, step == 120)

        # Train the model on the updated Q-values
        model.train_on_batch(state.reshape((1, 11, 11, 1)), Q_values)

        total_reward += reward
        state = next_state

    # Decay epsilon for exploration-exploitation trade-off
    epsilon *= epsilon_decay
    epsilon = max(min_epsilon, epsilon)

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Save the trained model for future use
# model.save('hex_agent_model.h5')
