import socket
import random
from copy import deepcopy
from math import sqrt, log
from time import time as clock
import sys
import os
# Add the project root directory to the sys.path to import Board and Colour
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../..', 'src'))
from Board import Board
from Colour import Colour
from queue import Queue

INF = float('inf')

class MCTSMeta:
    EXPLORATION = 0.5
    RAVE_CONST = 300
    # RANDOMNESS = 0.5
    # POOLRAVE_CAPACITY = 10
    # K_CONST = 10
    # A_CONST = 0.25
    # WARMUP_ROLLOUTS = 7

class Node:
    """
    Node for the MCST. Stores the move applied to reach this node from its parent,
    stats for the associated game position, children, parent and outcome 
    (outcome==none unless the position ends the game).
    Args:
        move:
        parent:
        N (int): times this position was visited
        Q (int): average reward (wins-losses) from this position
        Q_RAVE (int): times this move has been critical in a rollout
        N_RAVE (int): times this move has appeared in a rollout
        children (dict): dictionary of successive nodes
        outcome (int): If node is a leaf, then outcome indicates
                       the winner, else None
    """
    def __init__(self, move:tuple=None, parent:object=None, colour=None, moves=None):
        """
        Node for the MCTS. Stores the move applied to reach this node from its parent,
        stats for the associated game position, children, parent and outcome
        (outcome==none unless the position ends the game).
        Args:
            move:
            parent:
            N (int): times this position was visited
            Q (int): average reward (wins-losses) from this position
            Q_RAVE (int): times this move has been critical in a rollout
            N_RAVE (int): times this move has appeared in a rollout
            children (dict): dictionary of successive nodes
            outcome (int): If node is a leaf, then outcome indicates
                        the winner, else None
        """
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
        """
        Add a list of nodes to the children of this node.

        """
        for child in children:
            self.children[child.move] = child

    @property
    def value(self, explore: float = MCTSMeta.EXPLORATION, rave_const: float = MCTSMeta.RAVE_CONST) -> float:
        """
        Calculate the UCT value of this node relative to its parent, the parameter
        "explore" specifies how much the value should favor nodes that have
        yet to be thoroughly explored versus nodes that seem to have a high win
        rate.
        Currently explore is set to zero when choosing the best move to play so
        that the move with the highest win_rate is always chosen. When searching
        explore is set to EXPLORATION specified above.

        """
        # unless explore is set to zero, maximally favor unexplored nodes
        if self.N == 0:
            return 0 if explore == 0 else INF
        else:
            # rave valuation:
            alpha = max(0, (rave_const - self.N) / rave_const)
            UCT = self.Q / self.N + explore * sqrt(2 * log(self.parent.N) / self.N)
            AMAF = self.Q_RAVE / self.N_RAVE if self.N_RAVE != 0 else 0
            return (1 - alpha) * UCT + alpha * AMAF




class MCTSAgent:
    HOST = "127.0.0.1"
    PORT = 1234

    def __init__(self, board_size=11):
        """
        Constructor of MCTS Agent:
        Attributes:
        board: Game simulator that helps us to understand the game situation
        root (Node): Root of the tree search
        run_time (int): time per each run
        node_count (int): the whole nodes in tree
        turn_count (int): The number of rollouts for each search
        EXPLORATION (int): specifies how much the value should favor
                           nodes that have yet to be thoroughly explored versus nodes
                           that seem to have a high win rate.
        """

        # ----------------------------------------------------
        # |         Connects to the socket and jumps         |
        # |         to waiting for the start message         |
        # ----------------------------------------------------
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.HOST, self.PORT))

        # ----------------------------------------
        # |         Initialise Variables         |
        # ----------------------------------------
        self.board_size = board_size
        self.choices = []
        self.board = Board(board_size)
        self.colour = None
        self.run_time = 3
        self.node_count = 0 # keep track of the total number of nodes in the search tree that have been created and explored
        self.ifFirstStep = True
        self.root = Node()

    def run(self):
        """
        The agent run from here
        """
        while True:
            data = self.s.recv(1024)
            
            # To see the data received -- print(data)

            if not data:
                break
            
            # self.interpret_data(data) return False when receive "END" message
            if not self.interpret_data(data):
                break

        # print(f"MCTS agent {self.colour} terminated")



    def interpret_data(self, data):
        """
        Implement logic to interpret messages from the server
        Update the state of the agent based on the received data
        Returns False if the game ended, True otherwise.
        """

        messages = data.decode("utf-8").strip().split(";")

        # Close terminal when receive "END" message
        if (messages[0] == "END"):
            return False

        elif (messages[0] == "START"):
            self.colour = Colour.RED if messages[2] == "R" else Colour.BLUE
            self.board_size = int(messages[1])
            self.board = Board(self.board_size)
            self.choices = [(i, j) for i in range(self.board_size) for j in range(self.board_size)]
            
            # If the agent is RED, make the first move
            if self.colour == Colour.RED:
                if self.board_size >= 3:
                    self.s.sendall(bytes(f"0,2\n", "utf-8"))
                    self.ifFirstStep = False
                    self.choices.remove((0,2))
                else:
                    self.s.sendall(bytes(f"0,1\n", "utf-8"))
                    self.ifFirstStep = False
                    self.choices.remove((0,1))
        
        elif (messages[0] == "CHANGE"):
            self.board = Board.from_string(messages[2], board_size=self.board_size, bnf=True)

            if (messages[3]) == "END":
                return False
            
            elif messages[1] == "SWAP":
                self.colour = Colour.opposite(self.colour)
                if messages[3] == Colour.get_char(self.colour):
                    self.search()

            elif messages[3] == Colour.get_char(self.colour):
                opponent_move = tuple(map(int, messages[1].split(',')))
                if (self.ifFirstStep): # First step to decide if to swap
                    self.if_swap(opponent_move)
                # print(f"ALL CHOICES: {self.choices}, OPPONENT MOVE: {opponent_move}")
                else:
                    self.choices.remove(opponent_move)
                    self.search()

        # print(f"INTERPRET DATA: \n colour: {self.colour} \n board_size: {self.board_size} \n All Posible Moves: {self.choices} \n Board: {self.board.print_board(bnf=True)}")
        # print("END Interpret data\n")
        # print(self.run_time)
        # print(self.node_count)
        
        return True

    def if_swap(self,opponent_move):
        self.ifFirstStep = False
        self.choices.remove(opponent_move)
        badFirstStep = [(1,1),(3,1),(4,1),(5,1),(6,1),(7,1),(8,1),(9,1),(10,1),(2,5),
            (9,9),(7,9),(6,9),(5,9),(4,9),(3,9),(2,9),(1,9),(0,9),(8,5)] 
        if (opponent_move in badFirstStep) or (opponent_move[1] == 0) or (opponent_move[1] == (self.board_size - 1)):
            self.search()
        else:
            self.s.sendall(bytes("SWAP\n", "utf-8"))

    # SEARCH
    def search(self) -> None:
        """
        Return the best move according to the current tree.
        """
        
        end_time = clock() + self.run_time
        
        # Initialise root node
        self.root = Node(move=None, parent=None, colour=self.colour, moves=self.choices)
        # self.expand(self.root, self.choices, self.colour)
        # print(f"ROOT {self.root}: \n colour = {self.root.colour} \n move = {self.root.move} \n parent = {self.root.parent} \n children = {self.root.children} \n moves = {self.root.moves}")

        # Perform MCTS until the time limit is reached
        while clock() < end_time:
            # Selection & Expansion: Choose and expand a node to explore based on UCT values -- this should be the opponent turn
            selected_node, updated_board = self.select_node()
            # print(f"Selected Node: {selected_node}")
            # print(f"Colour: {selected_node.colour}")
            # print(f"Updated Board: {updated_board.print_board(bnf=True)}")
            # print(f"Selected moves: {selected_node.move}")
            # print(f"Moves Available: {selected_node.moves}")
            
            turn = Colour.opposite(selected_node.colour)

            # Simulation: Simulate a random playout from the new node
            outcome, red_rave_pts, blue_rave_pts = self.roll_out(selected_node, updated_board, turn)

            self.backup(selected_node, turn, outcome, red_rave_pts, blue_rave_pts)

        best_move = self.best_move()
        # print(f"Best_Move: {best_move} \n")
        self.choices.remove(best_move)

        # Send the chosen move to the server
        # print("===============SENT MESSAGE====================")
        self.send_move(best_move)


    def select_node(self) -> tuple:
        # print("========SELECTION============")
        """
        Select a node in the tree to preform a single simulation from.
        """
        node = self.root
        state = deepcopy(self.board)

        while node.children:
            # print("===========WHILE===============")
            # print("while", min(node.children.values(), key=lambda n:n.value).value)
            max_value = max(node.children.values(), key=lambda n:n.value).value
            max_nodes = [n for n in node.children.values() if n.value == max_value]
            node = random.choice(max_nodes)
            state.set_tile_colour(node.move[0], node.move[1], node.colour)
            # print(node.move[0], node.move[1], node.colour)
            # print(node.moves)

            #  if some child node has not been explored select it before expanding
            # other children
            if node.N == 0:
                # print(state.print_board(bnf=False))
                return node, state

        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        # print("========EXPAND===========")
        if self.expand(node, node.moves, node.colour):
            node = random.choice(list(node.children.values()))
            state.set_tile_colour(node.move[0], node.move[1], node.colour)
            # print(node.move[0], node.move[1], node.colour)
            # print(node.moves)
        # print(state.print_board(bnf=False))
        return node, state


    @staticmethod
    def expand(parent: Node, moves, colour) -> bool:
        # print("=========EXPANSION=========")
        """
        Generate the children of the passed "parent" node based on the available
        moves in the passed gamestate and add them to the tree.

        Returns:
            object:
        """
        children = []

        for move in moves:
            updated_moves = moves.copy()
            updated_moves.remove(move)

            # Determine the color for the child node
            child_colour = Colour.opposite(colour) if parent.N > 0 else colour
            # print(f"EXPAND COLOUR: {child_colour}")

            children.append(Node(move, parent, child_colour, updated_moves))

        parent.add_children(children)
        
        # print(f"Possible Number of Moves = {len(children)} : {children} \n")
        return True

    @staticmethod
    def roll_out(node: Node, state: Board, turn) -> tuple:
        """
        Simulate a random game except that we play all known critical
        cells first, return the winning player and record critical cells at the end.

        """
        # Get a list of all possible moves in current board of the game
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
        """
        Update the node statistics on the path from the passed node to root to reflect
        the outcome of a randomly simulated playout.
        """
        # note that reward is calculated for player who just played
        # at the node and not the next player to play
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
            # print(node.N, node.Q)

            turn = Colour.BLUE if turn == Colour.RED else Colour.RED
            reward = -reward
            node = node.parent

    def best_move(self) -> tuple:
        """
        Return the best move according to the current tree.
        Returns:
            best move in terms of the most simulations number unless the game is over
        """
        if self.board.has_ended():
            return 

        # choose the move of the most simulated node breaking ties randomly
        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        bestchild = random.choice(max_nodes)
        return bestchild.move

    def send_move(self, move):
        self.s.sendall(bytes(f"{move[0]},{move[1]}\n", "utf-8"))

if __name__ == "__main__":
    agent = MCTSAgent()
    agent.run()
