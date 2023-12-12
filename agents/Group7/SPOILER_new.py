import random
from copy import deepcopy
from math import sqrt, log
from time import time as clock
from Board import Board
from Colour import Colour

INF = float('inf')


class MCTSMeta:
    EXPLORATION = 0.5
    RAVE_CONST = 300


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
    def value(self, explore: float = MCTSMeta.EXPLORATION, rave_const: float = MCTSMeta.RAVE_CONST) -> float:
        if self.N == 0:
            return 0 if explore == 0 else INF
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

    def play_out(self, board, color):
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
        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        bestchild = random.choice(max_nodes)
        return bestchild.move
