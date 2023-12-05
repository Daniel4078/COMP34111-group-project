import socket
from random import choice
import model
from Board import Board
from Colour import Colour
import numpy as np


class Ouragent():
    HOST = "127.0.0.1"
    PORT = 1234

    def __init__(self, board_size=11):
        self.s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self.s.connect((self.HOST, self.PORT))
        self.board_size = board_size
        self.board = []
        self.colour = ""
        self.turn_count = 0

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
                self.board = [
                    [0] * self.board_size for i in range(self.board_size)]
                if self.colour == "R":
                    self.make_move()
            elif s[0] == "END":
                return True
            elif s[0] == "CHANGE":
                if s[3] == "END":
                    return True
                elif s[1] == "SWAP":
                    self.colour = self.opp_colour()
                    if s[3] == self.colour:
                        self.make_move()
                elif s[3] == self.colour:
                    action = [int(x) for x in s[1].split(",")]
                    self.board[action[0]][action[1]] = self.opp_colour()
                    self.make_move()
        return False

    def swap_map(self):
        # if position is in the center according to map then swap
        # else not swap
        return False

    def make_move(self):
        # run a alpha beta prunning minimax search that use neural network as heuristic provider
        # print(f"{self.colour} making move")
        if self.colour == "B" and self.turn_count == 0:
            if self.swap_map():  # use existing research results to decide swap or not
                self.s.sendall(bytes("SWAP\n", "utf-8"))
                self.turn_count += 1
                return
        if self.colour == "R" and self.turn_count == 0:
            # use existing research result to choose a node that would take the longest to win if opponent swap
            # aka a node that is at the boundary of first moves that are winning and not
            good_choices=[[1,2],[2,0],[3,0],[5,0],[6,0],[7,0],[8,0],[10,0],[2,5],[1,8],[0,10]]
            pos = choice(good_choices)
            if choice([0, 1]) == 1:
                pos = [pos[1], pos[0]]
            self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
            self.board[pos[0]][pos[1]] = self.colour
            self.turn_count += 1
            return
        depth = 4
        best_score, pos = self.minimax(self.board, depth, True, float('-inf'), float('inf'))
        pos = []
        self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
        self.board[pos[0]][pos[1]] = self.colour
        self.turn_count += 1

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        if depth == 0 or board.has_ended():
            return self.evaluate_board(board), None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in self.get_possible_moves(board, self.colour):
                board_copy = self.make_move_copy(board, move, self.colour)
                eval, _ = self.minimax(board_copy, depth - 1, False, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in self.get_possible_moves(board):
                board_copy = self.make_move_copy(board, move, self.opp_colour())
                eval, _ = self.minimax(board_copy, depth - 1, True, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate_board(self, tiles):  # TODO: adoptation to different board sizes
        condense = Board(board_size=6)
        for a in range(6):
            for b in range(6):
                part = []
                for i in range(6):
                    part.append(tiles[a + i][b:b + 6])
                state = model.board_to_state(part)
                Q_values = model.predict(state.reshape((1, 6, 6, 1)))
                if np.argmax(Q_values[0]) > 0:  # TODO: 优势劣势的分界线在哪？
                    condense.set_tile_colour(a, b, self.colour)
                else:
                    condense.set_tile_colour(a, b, self.opp_colour())
        state = model.board_to_state(condense.get_tiles())
        Q_values = model.predict(state.reshape((1, 6, 6, 1)))
        return np.argmax(Q_values[0])

    def get_possible_moves(self, board):
        moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x][y] == 0:
                    moves.append((x, y))
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
