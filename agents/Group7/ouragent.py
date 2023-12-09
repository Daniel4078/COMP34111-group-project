import socket
import random
import sys
import numpy as np
from tensorflow import keras
import time
sys.path.append(r"C:/Users/ttt/Desktop/COMP34111-group-project/src")
from Board import Board
from Colour import Colour

class Ouragent():
    HOST = "127.0.0.1"
    PORT = 1234

    def __init__(self, board_size=11):
        self.s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self.s.connect((self.HOST, self.PORT))
        self.board_size = board_size
        self.board = Board(self.board_size)
        self.colour = ""
        self.player_num = 0
        self.turn_count = 0
        self.last_move = None
        self.eva_model =  keras.models.load_model('board_evaluation_model.keras')
        self.step_model = keras.models.load_model('hex_agent_model.keras')

    def run(self):
        while True:
            data = self.s.recv(1024)
            if not data:
                break
            # print(f"{self.colour} {data.decode('utf-8')}", end="")
            if self.interpret_data(data):
                break
        # print(f"Naive agent {self.colour} terminated")

    def interpret_data(self, data):
        messages = data.decode("utf-8").strip().split("\n")
        messages = [x.split(";") for x in messages]
        # print(messages)
        for s in messages:
            if s[0] == "START":
                self.board_size = int(s[1])
                self.colour = s[2]
                self.board = Board(self.board_size)
                
                if self.colour == "R":
                    self.player_num = 1 
                    self.make_move()
                else:
                    self.player_num = 2
                    
            elif s[0] == "END":
                return True
            elif s[0] == "CHANGE":
                if s[3] == "END":
                    return True
                elif s[1] == "SWAP":
                    self.colour = self.opp_colour()
                    self.player_num = 2
                    if s[3] == self.colour:
                        self.make_move()
                elif s[3] == self.colour:
                    action = [int(x) for x in s[1].split(",")]
                    print(action)
                    self.board.set_tile_colour(action[0], action[1], Colour.from_char(self.opp_colour()))
                    # print("current")
                    print(self.board.print_board())
                    self.last_move = action
                    self.make_move()
        return False

    def swap_map(self):
        x = self.last_move[0]
        y = self.last_move[1]
        if 1 < x < 9 and 0 < y < 10:
            return True
        if x == 9 and (y == 0 or y == 1):
            return True
        if x == 1 and (y == 9 or y == 10):
            return True
        if (x == 10 and y == 0) or (x == 0 and y == 10):
            return True
        return False

    def tile_to_state(self, tile):
        colour = tile.get_colour()
        if colour == Colour.RED:
            return 1  # Assuming RED represents Player 1
        elif colour == Colour.BLUE:
            return 2  # Assuming BLUE represents Player 2
        else:
            return 0  # Assuming None or another value represents an empty tile
    
    
    def board_to_state(self, board_tiles):
        return np.array([[self.tile_to_state(tile) for tile in row] for row in board_tiles])

    def make_move(self):
        # print('make move')
        # print(f"{self.colour} making move")
        if self.colour == "B" and self.turn_count == 0:
            if self.swap_map():  # use existing research results to decide swap or not
                self.s.sendall(bytes("SWAP\n", "utf-8"))
                self.turn_count += 1
                return
        if self.colour == "R" and self.turn_count == 0:
            # use existing research result to choose a node that would take the longest to win if opponent swap
            # aka a node that is at the boundary of first moves that are winning and not
            good_choices = [[1, 2], [2, 0], [3, 0], [5, 0], [6, 0], [7, 0], [8, 0], [10, 0], [2, 5], [1, 8], [0, 10]]
            pos = random.choice(good_choices)
            if random.choice([0, 1]) == 1:
                pos = [pos[1], pos[0]]
            self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
            self.board.set_tile_colour(pos[0], pos[1], Colour.from_char(self.colour))
            self.last_move = pos
            self.turn_count += 1
            return
        # print("start minimax")
        best_score, pos = self.minimax(self.board, 4, True, float('-inf'), float('inf'))
        self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
        self.board.set_tile_colour(pos[0], pos[1], Colour.from_char(self.colour))
        print(self.board.print_board())
        self.last_move = pos
        self.turn_count += 1

    def make_move_copy(self, board, move, colour):
        temp = board.print_board()
        board_copy = Board.from_string(temp)
        board_copy.set_tile_colour(move[0], move[1], Colour.from_char(colour))
        return board_copy

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        if depth == 0 or board.has_ended():
            # print('evaluate')
            return self.evaluate_board(board.get_tiles(), maximizing_player), None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in self.get_good_moves(board.get_tiles()):
                newboard= self.make_move_copy(board, move, self.colour)
                eval, _ = self.minimax(newboard, depth - 1, False, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            # print(best_move)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in self.get_good_moves(board.get_tiles()):
                newboard= self.make_move_copy(board, move, self.opp_colour())
                eval, _ = self.minimax(newboard, depth - 1, True, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            # print(best_move)
            return min_eval, best_move

    # def evaluate_board(self, tiles, maximizing_player):  # TODO: adoptation to different board sizes
    #     print('start evaluate')
    #     condense = Board(board_size=6)
    #     for a in range(6):
    #         for b in range(6):
    #             part = []
    #             for i in range(6):
    #                 part.append(tiles[a + i][b:b + 6])
    #             state = self.board_to_state(part).reshape(1, 6, 6, 1)
    #             state_eval = np.append(state, np.full((1, state.shape[1], state.shape[2], state.shape[3]), self.player_num), axis=0)
    #             score = self.eva_model.predict(state_eval.reshape(1, 2, 6, 6, 1), verbose=0)
    #             # print('end predict')
    #             if score > 0.2:  # TODO: 优势劣势的分界线在哪？
    #                 condense.set_tile_colour(a, b, Colour.from_char(self.colour))
    #             elif score < -0.2:
    #                 condense.set_tile_colour(a, b, Colour.from_char(self.opp_colour()))
    #             # print(str(a)+" "+str(b))
    #     # print('done small board')
    #     state = self.board_to_state(condense.get_tiles()).reshape(1, 6, 6, 1)
    #     state_eval = np.append(state, np.full((1, state.shape[1], state.shape[2], state.shape[3]), self.player_num), axis=0)
    #     score = self.eva_model.predict(state_eval.reshape(1, 2, 6, 6, 1), verbose=0)
    #     if maximizing_player:
    #         score = -score
    #     return score
    
    def evaluate_board(self, tiles, maximizing_player):
        # Convert the entire board to the model's expected input format
        state = self.board_to_state(tiles).reshape(1, 11, 11, 1)
        
        # print(state.reshape(11,11))

        # Include player information if necessary
        state_eval = np.append(state, np.full((1, state.shape[1], state.shape[2], state.shape[3]), self.player_num), axis=0)

        # Use the model to predict the score of the current board state
        score = self.eva_model.predict(state_eval.reshape(1, 2, 11, 11, 1), verbose=0)
        
        if maximizing_player:
            score = -score

        return score

    def get_good_moves(self, tiles):
        state = self.board_to_state(tiles).reshape(1, 11, 11, 1)
        state_step = np.append(state, np.full((1, state.shape[1], state.shape[2], state.shape[3]), self.player_num), axis=0)
        Q_values = self.step_model.predict(state_step.reshape((1, 2, 11, 11, 1)), verbose=0)
        indexes = np.argsort(Q_values[0])[::-1]


        moves = []
        i = 0
        index = 0
        while i < 4:
            position = indexes[index]
            x, y = divmod(position, 11)
            if tiles[x][y].get_colour() is None:
                moves.append((x, y))
                i += 1
            index += 1
        print(moves)
        return moves

    def opp_colour(self):
        if self.colour == "R":
            return "B"
        elif self.colour == "B":
            return "R"
        else:
            return "None"


if (__name__ == "__main__"):
    agent = Ouragent()
    agent.run()
