import socket
import random
import numpy as np
from Board import Board
from Colour import Colour
import torch
import torch.nn as nn


class SkipLayerBias(nn.Module):

    def __init__(self, channels, reach, scale=1):
        super(SkipLayerBias, self).__init__()
        self.conv = nn.Conv2d(channels, channels,
                              kernel_size=reach * 2 + 1, padding=reach, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.scale = scale

    def forward(self, x):
        return self.swish(x + self.scale * self.bn(self.conv(x)))

    def swish(self, x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, board_size, layers, intermediate_channels, reach, export_mode):
        super(Conv, self).__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(2, intermediate_channels,
                              kernel_size=2 * reach + 1, padding=reach - 1)
        self.skiplayers = nn.ModuleList(
            [SkipLayerBias(intermediate_channels, 1) for idx in range(layers)])
        self.policyconv = nn.Conv2d(
            intermediate_channels, 1, kernel_size=2 * reach + 1, padding=reach, bias=False)
        self.bias = nn.Parameter(torch.zeros(board_size ** 2))
        self.export_mode = export_mode

    def forward(self, x):
        x_sum = torch.sum(x[:, :, 1:-1, 1:-1],
                          dim=1).view(-1, self.board_size ** 2)
        x = self.conv(x)
        for skiplayer in self.skiplayers:
            x = skiplayer(x)
        if self.export_mode:
            return self.policyconv(x).view(-1, self.board_size ** 2) + self.bias
        illegal = x_sum * torch.exp(torch.tanh((x_sum.sum(dim=1) - 1) * 1000)
                                    * 10).unsqueeze(1).expand_as(x_sum) - x_sum
        return self.policyconv(x).view(-1, self.board_size ** 2) + self.bias - illegal


class NoSwitchWrapperModel(nn.Module):
    def __init__(self, model):
        super(NoSwitchWrapperModel, self).__init__()
        self.board_size = model.board_size
        self.internal_model = model

    def forward(self, x):
        illegal = 1000 * torch.sum(x[:, :, 1:-1, 1:-1],
                                   dim=1).view(-1, self.board_size ** 2)
        return self.internal_model(x) - illegal


class RotationWrapperModel(nn.Module):
    def __init__(self, model, export_mode):
        super(RotationWrapperModel, self).__init__()
        self.board_size = model.board_size
        self.internal_model = model
        self.export_mode = export_mode

    def forward(self, x):
        if self.export_mode:
            return self.internal_model(x)
        x_flip = torch.flip(x, [2, 3])
        y_flip = self.internal_model(x_flip)
        y = torch.flip(y_flip, [1])
        return (self.internal_model(x) + y) / 2


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
        self.current = 0
        self.maxcurrent = 0
        self.last_move = None
        self.neighbor_patterns = (
            (-1, 0), (0, -1), (-1, 1), (0, 1), (1, 0), (1, -1))
        self.model = self.load_model('1111.pt')

    def load_model(self, model_file, export_mode=False):
        checkpoint = torch.load(model_file, map_location="cpu")
        model = self.create_model(checkpoint['config'], export_mode)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        torch.no_grad()
        return model

    def create_model(self, config, export_mode=False):
        board_size = config.getint('board_size')
        switch_model = config.getboolean('switch_model')
        rotation_model = config.getboolean('rotation_model')

        model = Conv(
            board_size=board_size,
            layers=config.getint('layers'),
            intermediate_channels=config.getint('intermediate_channels'),
            reach=config.getint('reach'),
            export_mode=export_mode
        )

        if not switch_model:
            model = NoSwitchWrapperModel(model)

        if rotation_model:
            model = RotationWrapperModel(model, export_mode)

        return model

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
                    self.board.set_tile_colour(
                        action[0], action[1], Colour.from_char(self.opp_colour()))
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

    def get_empty(self, state):
        indices = []
        for x in range(self.board_size + 4):
            for y in range(self.board_size + 4):
                if (state[0, x, y] == 0 and state[1, x, y] == 0):
                    indices.append((x, y))
        return indices

    def get_sp_state(self, state):
        sp_state1 = np.zeros((self.board_size, self.board_size), dtype=bool)
        sp_state2 = np.zeros((self.board_size, self.board_size), dtype=bool)
        np.place(sp_state1, state == 2, 1)
        np.place(sp_state2, state == 1, 1)
        return np.array([np.pad(sp_state1, 2, constant_values=1), np.pad(sp_state2, 2, constant_values=1)])

    def neighbors(self, cell):
        x = cell[0]
        y = cell[1]
        return [(n[0] + x, n[1] + y) for n in self.neighbor_patterns
                if
                (0 <= n[0] + x and n[0] + x < self.board_size + 4 and 0 <= n[1] + y and n[1] + y < self.board_size + 4)]

    def fill_connect(self, state, cell, color, checked):
        checked[cell] = True
        connected = set()
        for n in self.neighbors(cell):
            if (not checked[n]):
                if (state[color, n[0], n[1]]):
                    connected = connected | self.fill_connect(
                        state, n, color, checked)
                elif (not state[1 - color, n[0], n[1]]):
                    connected.add(n)
        return connected

    def get_connections(self, state, color, empty, checked):
        connections = {cell: set() for cell in empty}
        for cell in empty:
            for n in self.neighbors(cell):
                if (not checked[n]):
                    if (state[color, n[0], n[1]]):
                        connected_set = self.fill_connect(
                            state, n, color, checked)
                        for c1 in connected_set:
                            for c2 in connected_set:
                                connections[c1].add(c2)
                                connections[c2].add(c1)
                    elif (not state[1 - color, n[0], n[1]]):
                        connections[cell].add(n)
        return connections

    def resistance(self, state, empty, color):
        if self.board.has_ended():
            if self.board.get_winner() == color:
                return np.zeros((self.board_size + 4, self.board_size + 4)), float("inf")
            else:
                return np.zeros((self.board_size + 4, self.board_size + 4)), 0
        index_to_location = empty
        num_empty = len(empty)
        location_to_index = {index_to_location[i]: i for i in range(num_empty)}
        if color == "R":
            n_color = 1
        else:
            n_color = 0
        I = np.zeros(num_empty)
        G = np.zeros((num_empty, num_empty))

        checked = np.zeros((self.board_size + 4, self.board_size + 4), dtype=bool)
        source_connected = self.fill_connect(state, (0, 0), n_color, checked)
        for n in source_connected:
            j = location_to_index[n]
            I[j] += 1
            G[j, j] += 1

        dest_connected = self.fill_connect(
            state, (self.board_size + 3, self.board_size + 3), n_color, checked)
        for n in dest_connected:
            j = location_to_index[n]
            G[j, j] += 1
        adjacency = self.get_connections(
            state, n_color, index_to_location, checked)
        for c1 in adjacency:
            j = location_to_index[c1]
            for c2 in adjacency[c1]:
                i = location_to_index[c2]
                G[i, j] -= 1
                G[i, i] += 1
        try:
            V = np.linalg.solve(G, I)
        except np.linalg.linalg.LinAlgError:
            V = np.linalg.lstsq(G, I)[0]
        C = 0
        Il = np.zeros((self.board_size + 4, self.board_size + 4))
        for i in range(num_empty):
            if index_to_location[i] in source_connected:
                Il[index_to_location[i]] += abs(V[i] - 1) / 2
            if index_to_location[i] in dest_connected:
                Il[index_to_location[i]] += abs(V[i]) / 2
            for j in range(num_empty):
                if (i != j and G[i, j] != 0):
                    Il[index_to_location[i]] += abs(G[i, j] * (V[i] - V[j])) / 2
                    if (index_to_location[i] in source_connected and
                            index_to_location[j] not in source_connected):
                        C += -G[i, j] * (V[i] - V[j])
        return Il, C

    def make_move(self):
        if self.colour == "B" and self.turn_count == 0:
            if self.swap_map():
                self.s.sendall(bytes("SWAP\n", "utf-8"))
                self.turn_count += 1
                return
        if self.colour == "R" and self.turn_count == 0:
            good_choices = [[1, 2], [2, 0], [3, 0], [5, 0], [6, 0], [
                7, 0], [8, 0], [10, 0], [2, 5], [1, 8], [0, 10]]
            pos = random.choice(good_choices)
            if random.choice([0, 1]) == 1:
                pos = [pos[1], pos[0]]
            self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
            self.board.set_tile_colour(
                pos[0], pos[1], Colour.from_char(self.colour))
            self.last_move = pos
            self.turn_count += 1
            return
        _, pos = self.minimax(self.board, 1, True, float('-inf'), float('inf'))
        self.current = self.evaluate_board(
            self.board_to_state(self.board.get_tiles()), True)
        self.maxcurrent = max(self.current, self.maxcurrent)
        print(self.maxcurrent)
        self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
        self.board.set_tile_colour(
            pos[0], pos[1], Colour.from_char(self.colour))
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
            for move in self.pick_moves(board, self.colour):
                newboard = self.make_move_copy(board, move, self.colour)
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
            for move in self.pick_moves(board, self.opp_colour()):
                newboard = self.make_move_copy(board, move, self.opp_colour())
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
            _, s = self.resistance(temp, self.get_empty(temp), c)
            return s
        else:
            c = self.opp_colour()
            _, s = self.resistance(temp, self.get_empty(temp), c)
            return -s

    def pick_moves_sp(self, board, colour):
        pass

    def pick_moves(self, board, colour):
        state = self.board_to_state(board.get_tiles())
        sp_state1 = np.zeros((self.board_size, self.board_size))
        sp_state2 = np.zeros((self.board_size, self.board_size))
        np.place(sp_state1, state == 1, 1)
        np.place(sp_state2, state == 2, 1)
        boards = torch.tensor([np.pad(sp_state1, 1, constant_values=1), np.pad(
            sp_state2, 1, constant_values=1)], dtype=torch.float)
        current_boards_tensor = boards.unsqueeze(0)
        with torch.no_grad():
            outputs_tensor = self.model(current_boards_tensor)
        Q_values = np.array(outputs_tensor)[0]
        indexes = np.argsort(Q_values)[::-1]
        temp = []
        for index in indexes[:1]:
            x, y = divmod(index, 11)
            temp.append([x, y])
        return temp

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
