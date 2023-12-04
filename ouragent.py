import socket
from random import choice


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
        

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        if depth == 0 or self.is_game_over(board):
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
            for move in self.get_possible_moves(board, self.opp_colour()):
                board_copy = self.make_move_copy(board, move, self.opp_colour())
                eval, _ = self.minimax(board_copy, depth - 1, True, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move
        
    def evaluate_board(self, board):
        # 评估当前胜率
            return 0
        
    def is_game_over(self, board):
        # 实现游戏是否结束
            return False
        
        # 获取可能的移动列表    
    def get_possible_moves(self, board, colour):
            moves = []
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if board[x][y] == 0:  
                        moves.append((x, y))
            return moves

    # 创建执行移动后的棋盘副本
    def make_move_copy(self, board, move, colour):
        board_copy = [row[:] for row in board]
        board_copy[move[0]][move[1]] = colour
        return board_copy


    def make_move(self):
        # 确定搜索深度
        depth = 3

        # 进行 minimax 搜索并选择最佳移动
        best_score, best_move = self.minimax(self.board, depth, True, float('-inf'), float('inf'))

        # 发送最佳移动
        self.s.sendall(bytes(f"{best_move[0]},{best_move[1]}\n", "utf-8"))

        # 更新棋盘
        self.board[best_move[0]][best_move[1]] = self.colour

        self.turn_count += 1

    # def make_move(self):
    #     # run a alpha beta prunning minimax search that use neural network as heuristic provider
    #     # print(f"{self.colour} making move")
    #     if self.colour == "B" and self.turn_count == 0:
    #         if :#use existing research results to decide
    #             self.s.sendall(bytes("SWAP\n", "utf-8"))
    #         else:
    #             #use existing research results to get a position that take the longest to lose/win
    #             pos = []
    #             self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
    #             self.board[pos[0]][pos[1]] = self.colour
    #     else:
    #         # put the tree search here
    #         pos = []
    #         self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
    #         self.board[pos[0]][pos[1]] = self.colour
    #     self.turn_count += 1

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
