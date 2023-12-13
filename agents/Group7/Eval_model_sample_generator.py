import random
import numpy as np
import csv
import time
import SPOILER_new
from Game import Game
from Colour import Colour

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


# Training parameters
num_episodes = 10
csv_file_path = 'board_evaluation.csv'

for episode in range(num_episodes):
    # Initialization
    game = Game(board_size=11)
    player1 = random.choice([Colour.RED, Colour.BLUE])
    # player1_color = Colour.RED
    # player1_color = Colour.BLUE

    if player1 == Colour.RED:
        player1_color = "R"
        player2_color = "B"
        player2 = Colour.BLUE
    else:
        player1_color = "B"
        player2_color = "R"
        player2 = Colour.RED

    start = True

    # To store every board state during one game
    States_eval = []
    States2_eval = []

    player1_step = 0
    player2_step = 0

    player = SPOILER_new.MCTSAgent()

    # Set timeer
    startTime = time.time()
    while True:
        # Let Red starts first
        if player1 == Colour.RED or start == False:
            player1_step +=1
            print(f"Player 1 is taking step {player1_step}...")

            action = player.play_out(game.get_board(), player2_color)

            game.get_board().set_tile_colour(action[0], action[1], player1)

            state = board_to_state(game.get_board().get_tiles())
            state = state.reshape((1, 11, 11, 1))

            States_eval.append(state)

            if game.get_board().has_ended():
                break

        start = False
        # Player2 
        player2_step +=1
        print(f"Player 2 is taking step {player2_step}...")

        action2 = player.play_out(game.get_board(), player2_color)

        game.get_board().set_tile_colour(action2[0], action2[1], player2)

        state = board_to_state(game.get_board().get_tiles())
        state = state.reshape((1, 11, 11, 1))

        States2_eval.append(state)

        if game.get_board().has_ended():
            break
    run_time = time.time() - startTime

    # Train the step prediction model
    board_scores = []
    for move_num in range(len(States_eval)):
        if game.get_board().get_winner() == player1:
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
        if game.get_board().get_winner() == player1:
            board_scores.insert(0, -1 * (0.86 ** move_num))
        else:
            board_scores.insert(0, 1 * (0.86 ** move_num))

    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for move_num in range(len(States2_eval)):
            # Save the state and board score to the CSV file
            csv_writer.writerow(list(States2_eval[move_num]) + [board_scores[move_num]])

    print(state.reshape(11, 11))
    print(f"Episode: {episode + 1}")
    print(f"Runing time: {run_time}, Total steps: {player1_step + player2_step}")
    print(f"Winner: {game.get_board().get_winner()}\n")