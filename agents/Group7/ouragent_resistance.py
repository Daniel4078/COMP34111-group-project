import socket
import random
import numpy as np
from Board import Board
from Colour import Colour
from copy import deepcopy
from math import sqrt, log
from time import time as clock

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
        self.neighbor_patterns = ((-1,0), (0,-1), (-1,1), (0,1), (1,0), (1,-1))
        self.model = MCTSAgent()


    def run(self):
        while True:
            data = self.s.recv(1024)
            if not data:
                break
            if self.interpret_data(data):
                break

    def interpret_data(self, data):
        messages = data.decode("utf-8").strip().split("\n")
        messages = [x.split(";") for x in messages]
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
                    self.board.set_tile_colour(action[0], action[1], Colour.from_char(self.opp_colour()))
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
            return 1
        elif colour == Colour.BLUE:
            return 2
        else:
            return 0
    
    
    def board_to_state(self, board_tiles):
        return np.array([[self.tile_to_state(tile) for tile in row] for row in board_tiles])
    
    
    def get_empty(self,state):
        indices = []
        for x in range(self.board_size+4):
            for y in range(self.board_size+4):
                if(state[0,x,y] == 0 and state[1,x,y] == 0):
                    indices.append((x,y))
        return indices

    def get_sp_state(self, state):
        sp_state1 = np.zeros((self.board_size, self.board_size), dtype=bool)
        sp_state2 = np.zeros((self.board_size, self.board_size), dtype=bool)
        np.place(sp_state1, state==2, 1)
        np.place(sp_state2, state==1, 1)
        return np.array([np.pad(sp_state1, 2, constant_values=1),np.pad(sp_state2, 2, constant_values=1)])

    def neighbors(self, cell):
        """
        Return list of neighbors of the passed cell.
        """
        x = cell[0]
        y = cell[1]
        return [(n[0] + x , n[1] + y) for n in self.neighbor_patterns\
            if (0 <= n[0] + x and n[0] + x < self.board_size+4 and 0 <= n[1] + y and n[1] + y < self.board_size+4)]
        
    
    def fill_connect(self, state, cell, color, checked):
        checked[cell] = True
        connected = set()
        for n in self.neighbors(cell):
            if(not checked[n]):
                if(state[color, n[0], n[1]]):
                    connected = connected | self.fill_connect(state, n, color, checked)
                elif(not state[1 - color, n[0], n[1]]):
                    connected.add(n)
        return connected


    def get_connections(self, state, color, empty, checked):
        connections = {cell:set() for cell in empty}
        for cell in empty:
            for n in self.neighbors(cell):
                if(not checked[n]):
                    if(state[color, n[0], n[1]]):
                        connected_set = self.fill_connect(state, n, color, checked)
                        for c1 in connected_set:
                            for c2 in connected_set:
                                connections[c1].add(c2)
                                connections[c2].add(c1)
                    elif(not state[1 - color,n[0],n[1]]):
                        connections[cell].add(n)
        return connections
    
    
    def resistance(self, state, empty, color):
        """
        Output a resistance heuristic score over all empty nodes:
            -Treat the west edge connected nodes as one source node with voltage 1
            -Treat east edge connected nodes as one destination node with voltage 0
            -Treat edges between empty nodes, or from source/dest to an empty node as resistors with conductance 1
            -Treat edges to blue nodes (except source and dest) as perfect conductors 左右
            -Treat edges to red nodes as perfect resistors 上下
        """
        
        if self.board.has_ended():
            if  self.board.get_winner() == color:
                return float("inf")
            else:
                return 0
            
        index_to_location = empty
        num_empty = len(empty)
        location_to_index = {index_to_location[i]:i for i in range(num_empty)}
        if color == "R" :
            n_color = 1
        else: 
            n_color = 0
        I = np.zeros(num_empty)
        G = np.zeros((num_empty, num_empty))

        checked = np.zeros((self.board_size+4, self.board_size+4), dtype=bool)
        source_connected = self.fill_connect(state, (0,0), n_color, checked)
        for n in source_connected:
            j = location_to_index[n]
            I[j] += 1
            G[j,j] += 1
            

        dest_connected = self.fill_connect(state, (self.board_size+3, self.board_size+3), n_color, checked)
        for n in dest_connected:
            j = location_to_index[n]
            G[j,j] +=1
        adjacency = self.get_connections(state, n_color, index_to_location, checked)
        for c1 in adjacency:
            j=location_to_index[c1]
            for c2 in adjacency[c1]:
                i=location_to_index[c2]
                G[i,j] -= 1
                G[i,i] += 1
        try:
            V = np.linalg.solve(G,I)
        except np.linalg.linalg.LinAlgError:
            V = np.linalg.lstsq(G,I)[0]
        V_board = np.zeros((self.board_size+4, self.board_size+4))
        for i in range(num_empty):
            V_board[index_to_location[i]] = V[i]
        C = 0
        for i in range(num_empty):
            for j in range(num_empty):
                if(i!=j and G[i,j] != 0):
                    if(index_to_location[i] in source_connected and
                    index_to_location[j] not in source_connected):
                        C+=-G[i,j]*(V[i] - V[j])
        return C

    def make_move(self):
        if self.colour == "B" and self.turn_count == 0:
            if self.swap_map():
                self.s.sendall(bytes("SWAP\n", "utf-8"))
                self.turn_count += 1
                return
        if self.colour == "R" and self.turn_count == 0:
            good_choices = [[1, 2], [2, 0], [3, 0], [5, 0], [6, 0], [7, 0], [8, 0], [10, 0], [2, 5], [1, 8], [0, 10]]
            pos = random.choice(good_choices)
            if random.choice([0, 1]) == 1:
                pos = [pos[1], pos[0]]
            self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
            self.board.set_tile_colour(pos[0], pos[1], Colour.from_char(self.colour))
            self.last_move = pos
            self.turn_count += 1
            return
        _, pos = self.minimax(self.board, 3, True, float('-inf'), float('inf'))
        self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
        self.board.set_tile_colour(pos[0], pos[1], Colour.from_char(self.colour))
        self.last_move = pos
        self.turn_count += 1

    def make_move_copy(self, board, move, colour):
        temp = board.print_board()
        board_copy = Board.from_string(temp)
        board_copy.set_tile_colour(move[0], move[1], Colour.from_char(colour))
        return board_copy

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        if depth == 0 or board.has_ended():
            return self.evaluate_board(self.board_to_state(board.get_tiles()), maximizing_player), None
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in self.model.get_moves(board, self.colour):
                newboard= self.make_move_copy(board, move, self.colour)
                eval, _ = self.minimax(newboard, depth - 1, False, alpha, beta)
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
            for move in self.model.get_moves(board, self.opp_colour()):
                newboard= self.make_move_copy(board, move, self.opp_colour())
                eval, _ = self.minimax(newboard, depth - 1, True, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move
    
    def evaluate_board(self, tiles, maximizing_player):
        temp = self.get_sp_state(tiles)
        if maximizing_player:
            c = self.colour
            return self.resistance(temp, self.get_empty(temp), c)
        else:
            c = self.opp_colour()
            return -self.resistance(temp, self.get_empty(temp), c)
 

    def opp_colour(self):
        if self.colour == "R":
            return "B"
        elif self.colour == "B":
            return "R"
        else:
            return "None"

class Node:
    def __init__(self, move: tuple = None, parent: object = None, colour=None, moves=None):
        self.move = move
        self.parent = parent
        self.moves = moves
        self.colour = colour
        self.N = 0  # times this position was visited
        self.Q = 0  # average reward (wins-losses) from this position
        self.Q_RAVE = 0
        self.N_RAVE = 0
        self.children = {}
        self.outcome = None

    def add_children(self, children: dict) -> None:
        for child in children:
            self.children[child.move] = child

    @property
    def value(self, explore = 0.5, rave_const = 300):
        if self.N == 0:
            return 0 if explore == 0 else float('inf')
        else:
            alpha = max(0, (rave_const - self.N) / rave_const)
            UCT = self.Q / self.N + explore * sqrt(2 * log(self.parent.N) / self.N)
            AMAF = self.Q_RAVE / self.N_RAVE if self.N_RAVE != 0 else 0
            return (1 - alpha) * UCT + alpha * AMAF


class MCTSAgent:
    def __init__(self, board_size=11):
        self.board_size = board_size
        self.choices = []
        self.board = Board(board_size)
        self.colour = None
        self.run_time = 3
        self.node_count = 0  # keep track of the total number of nodes in the search tree that have been created and explored
        self.ifFirstStep = True
        self.root = Node()

    def get_moves(self, board, color):
        self.colour = Colour.from_char(color)
        self.board = board
        temp = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if Colour.get_char(self.board.get_tiles()[i][j].get_colour()) == "0":
                    temp.append((i, j))
        self.choices = temp
        return self.search()

    # SEARCH
    def search(self) -> None:
        end_time = clock() + self.run_time
        self.root = Node(move=None, parent=None, colour=self.colour, moves=self.choices)
        while clock() < end_time:
            selected_node, updated_board = self.select_node()
            turn = Colour.opposite(selected_node.colour)
            outcome, red_rave_pts, blue_rave_pts = self.roll_out(selected_node, updated_board, turn)
            self.backup(selected_node, turn, outcome, red_rave_pts, blue_rave_pts)

        best_move = self.best_move()
        return list(best_move)

    def select_node(self) -> tuple:
        node = self.root
        state = deepcopy(self.board)

        while node.children:
            max_value = max(node.children.values(), key=lambda n: n.value).value
            max_nodes = [n for n in node.children.values() if n.value == max_value]
            node = random.choice(max_nodes)
            state.set_tile_colour(node.move[0], node.move[1], node.colour)
            if node.N == 0:
                return node, state

        if self.expand(node, node.moves, node.colour):
            node = random.choice(list(node.children.values()))
            state.set_tile_colour(node.move[0], node.move[1], node.colour)

        return node, state

    @staticmethod
    def expand(parent: Node, moves, colour) -> bool:
        children = []

        for move in moves:
            updated_moves = moves.copy()
            updated_moves.remove(move)
            child_colour = Colour.opposite(colour) if parent.N > 0 else colour
            children.append(Node(move, parent, child_colour, updated_moves))

        parent.add_children(children)

        return True

    @staticmethod
    def roll_out(node: Node, state: Board, turn) -> tuple:
        moves = node.moves.copy()

        while not state.has_ended():
            move = random.choice(moves)
            state.set_tile_colour(move[0], move[1], turn)
            moves.remove(move)
            turn = Colour.opposite(turn)

        red_rave_pts = []
        blue_rave_pts = []

        board_str = state.print_board(bnf=True)
        rows = board_str.split(',')
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                if cell == 'R':
                    red_rave_pts.append((i, j))
                elif cell == 'B':
                    blue_rave_pts.append((i, j))

        return state.get_winner(), red_rave_pts, blue_rave_pts

    def backup(self, node: Node, turn: int, outcome: int, red_rave_pts: list, blue_rave_pts: list) -> None:

        reward = -1 if outcome == turn else 1

        while node is not None:
            if turn == Colour.BLUE:
                for point in blue_rave_pts:
                    if point in node.children:
                        node.children[point].Q_RAVE += -reward
                        node.children[point].N_RAVE += 1
            else:
                for point in red_rave_pts:
                    if point in node.children:
                        node.children[point].Q_RAVE += -reward
                        node.children[point].N_RAVE += 1

            node.N += 1
            node.Q += reward

            turn = Colour.BLUE if turn == Colour.RED else Colour.RED
            reward = -reward
            node = node.parent

    def best_move(self):
        sorted_children = sorted(self.root.children.values(), key=lambda n: n.N, reverse=True)
        top_moves = sorted_children[:3]
        top_moves = [node.move for node in top_moves]
        return top_moves


if (__name__ == "__main__"):
    agent = Ouragent()
    agent.run()
