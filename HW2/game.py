"""
ChineseChecker is a class that represents the game logic for Chinese Checkers.

Attributes:
    size (int): The size of the board.
    piece_rows (int): The number of rows occupied by pieces at the start.
    board (Board): The game board.

Methods:
    __init__(self, size, piece_rows):
        Initializes the ChineseChecker with a board of given size and piece rows.

    startState(self):
        Resets the board and returns the initial state.

    isEnd(self, state, iter):
        Checks if the game has ended given the current state and iteration.

    actions(self, state):
        Returns a list of possible actions for the current player in the given state.

    opp_actions(self, state):
        Returns a list of possible actions for the opponent in the given state.

    player(self, state):
        Returns the current player from the state.

    succ(self, state, action):
        Returns the successor state after applying the given action for the current player.

    opp_succ(self, state, action, last_action):
        Returns the successor state after applying the given action for the opponent.
"""

import time
from board import Board
import copy


class ChineseChecker(object):

    def __init__(self, size, piece_rows):
        self.size = size
        self.piece_rows = piece_rows
        self.board = Board(self.size, self.piece_rows)

    def startState(self):
        self.board = Board(self.size, self.piece_rows)
        return (1, self.board)

    def isEnd(self, state, iter):
        return state[1].isEnd(iter)[0]

    def actions(self, state):
        action_list = []
        player = state[0]
        board = state[1]
        player_piece_pos_list = board.getPlayerPiecePositions(player)
        for pos in player_piece_pos_list:
            for adj_pos in board.adjacentPositions(pos):
                if board.isEmptyPosition(adj_pos):
                    action_list.append((pos, adj_pos))

        for pos in player_piece_pos_list:
            boardCopy = copy.deepcopy(board)
            boardCopy.board_status[pos] = 0
            for new_pos in boardCopy.getAllHopPositions(pos):
                if (pos, new_pos) not in action_list:
                    action_list.append((pos, new_pos))

        return action_list

    def opp_actions(self, state):
        action_list = []
        player = state[0]
        board = state[1]
        player_piece_pos_list = board.getPlayerPiecePositions(player)
        for pos in player_piece_pos_list:
            for adj_pos in board.adjacentPositions(pos):
                if board.isEmptyPosition(adj_pos):
                    action_list.append((pos, adj_pos))

        for pos in player_piece_pos_list:
            boardCopy = copy.deepcopy(board)
            boardCopy.board_status[pos] = 0
            for new_pos in boardCopy.getAllHopPositions(pos):
                if (pos, new_pos) not in action_list:
                    action_list.append((pos, new_pos))

        return action_list

    def player(self, state):
        return state[0]

    def succ(self, state, action):

        move_opp = False
        player = state[0]
        board = copy.deepcopy(state[1])
        board.board_status[action[1]] = board.board_status[action[0]]

        if str(action[1]) in self.board.player1_pos and board.board_status[action[1]] == 3 and player == 1:
            if self.board.player1_pos[str(action[1])] == False:
                move_opp = True

        elif str(action[1]) in self.board.player2_pos and board.board_status[action[1]] == 4 and player == 2:
            if self.board.player2_pos[str(action[1])] == False:
                move_opp = True

        board.board_status[action[0]] = 0

        return (3 - player, board, move_opp)

    def opp_succ(self, state, action, last_action):

        move_opp = False
        player = state[0]
        print(action)
        board = copy.deepcopy(state[1])
        board.board_status[action[1]] = board.board_status[action[0]]
        board.board_status[action[0]] = 0
        if 3 - player == 1:
            self.board.player1_pos[str(last_action[1])] = True
        elif 3 - player == 2:
            self.board.player2_pos[str(last_action[1])] = True

        return (player, board, move_opp)


# okay decompiling game.pyc
