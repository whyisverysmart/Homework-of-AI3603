"""
This module defines various agent classes for a game, including random agents, greedy agents.
You need to implement your own agent in the YourAgent class using minimax algorithms.

Classes:
    Agent: Base class for all agents.
    RandomAgent: Agent that selects actions randomly.
    SimpleGreedyAgent: Greedy agent that selects actions based on maximum vertical advance.
    YourAgent: Placeholder for user-defined agent.

Class Agent:
    Methods:
        __init__(self, game): Initializes the agent with the game instance.
        getAction(self, state): Abstract method to get the action for the current state.
        oppAction(self, state): Abstract method to get the opponent's action for the current state.

Class RandomAgent(Agent):
    Methods:
        getAction(self, state): Selects a random legal action.
        oppAction(self, state): Selects a random legal action for the opponent.

Class SimpleGreedyAgent(Agent):
    Methods:
        getAction(self, state): Selects an action with the maximum vertical advance.
        oppAction(self, state): Selects an action with the minimum vertical advance for the opponent.

Class YourAgent(Agent):
    Methods:
        getAction(self, state): Placeholder for user-defined action selection.
        oppAction(self, state): Placeholder for user-defined opponent action selection.
"""

import random, re, datetime
import board
from concurrent.futures import ProcessPoolExecutor, as_completed


class Agent(object):
    def __init__(self, game):
        self.game = game
        self.action = None

    def getAction(self, state):
        raise Exception("Not implemented yet")

    def oppAction(self, state):
        raise Exception("Not implemented yet")


class RandomAgent(Agent):

    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)

    def oppAction(self, state):
        legal_actions = self.game.actions(state)
        self.opp_action = random.choice(legal_actions)


class SimpleGreedyAgent(Agent):
    # a one-step-lookahead greedy agent that returns action with max vertical advance

    def getAction(self, state):

        legal_actions = self.game.actions(state)

        self.action = random.choice(legal_actions)

        player = self.game.player(state)
        if player == 1:
            max_vertical_advance_one_step = max([action[0][0] - action[1][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if action[0][0] - action[1][0] == max_vertical_advance_one_step]
        else:
            max_vertical_advance_one_step = max([action[1][0] - action[0][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if action[1][0] - action[0][0] == max_vertical_advance_one_step]
        self.action = random.choice(max_actions)

    def oppAction(self, state):
        legal_actions = self.game.actions(state)

        self.opp_action = random.choice(legal_actions)

        player = self.game.player(state)
        if player == 1:
            min_vertical_advance_one_step = min([action[0][0] - action[1][0] for action in legal_actions])
            min_actions = [action for action in legal_actions if action[0][0] - action[1][0] == min_vertical_advance_one_step]
        else:
            min_vertical_advance_one_step = min([action[1][0] - action[0][0] for action in legal_actions])
            min_actions = [action for action in legal_actions if action[1][0] - action[0][0] == min_vertical_advance_one_step]

        self.opp_action = random.choice(min_actions)


class YourAgent(Agent):

    def eval_state(self, state):
        """Evaluate the current state.
        Args:
            state: The current state of the board.
        Returns:
            total_point(float): The total point of the current state, lower is better for player 1, vise versa.
        """
        board = state[1]
        status = board.board_status
        row1 = 0
        row2 = 0
        total_point = 0
        special_bonus_ratio = 0.5   # Actually is 0.5 + 1
        normal_bonus_ratio = 1
        center_bonus_ratio = 0.5
        for key, value in status.items():
            if value == 0:
                continue
            elif value == 3:
                ### 1. Sum the row of piece
                ### 2. Bonus and Penalty:
                ###    - Should get into the right position
                ###    - Should move faster towards the right position
                row1 += key[0]
                if key[0] == 2:
                    total_point -= 100
                elif key[0] == 1:
                    total_point += 100
                else:
                    total_point += (key[0] - 2) * special_bonus_ratio
            elif value == 4:
                ### Similar as above
                row2 += key[0]
                if key[0] == 18:
                    total_point += 100
                elif key[0] == 19:
                    total_point -= 100
                else:
                    total_point -= (18 - key[0]) * special_bonus_ratio
            elif value == 1:
                ### 1. Sum the row of piece
                ### 2. Bonus and Penalty:
                ###    - Should maintain in center
                ###    - Should not occupy the special position
                row1 += key[0]
                total_point += abs(key[1] - min(key[0]+1, 21-key[0])/2) * center_bonus_ratio
                if key[0] == 2:
                    total_point += 100
            elif value == 2:
                ### Similar as above
                row2 += key[0]
                total_point -= abs(key[1] - min(key[0]+1, 21-key[0])/2) * center_bonus_ratio
                if key[0] == 18:
                    total_point -= 100

        ### Winning condition
        if row1 == 30 and (status[(2, 1)] == 3 and status[(2, 2)] == 3):
            return -100000
        if row2 == 170 and (status[(18, 1)] == 4 and status[(18, 2)] == 4):
            return 100000

        total_point += (row1 + row2) * normal_bonus_ratio
        return total_point
    
    def get_top_actions(self, state, player):
        """Returns the top 1/2 actions with max evaluation. Not needed when depth is 2.
        Args:
            state: The current state of the board.
            player: The current player.
        Returns:
            result(list): The list of top actions.
        """
        legal_actions = self.game.actions(state)
        if player == 1:
            result = sorted(legal_actions, key=lambda x: x[1][0]-x[0][0])
            return result[:int(len(result)/2)]
        else:
            result = sorted(legal_actions, key=lambda x: x[0][0]-x[1][0])
            return result[:int(len(result)/2)]


    def alphabeta(self, state, depth, a, b, player):
        """The minimax algorithm with alpha-beta pruning.
        Args:
            state: The current state of the board.
            depth: The depth of search.
            a: Alpha.
            b: Beta.
            player: The current player.
        Returns:
            float: alpha or beta.
        """
        if depth == 0:
            return self.eval_state(state)

        ### Player 2 wants the point to be as large as possible(Max player)
        if player == 2:
            top_actions = self.get_top_actions(state, 2)
            for act in top_actions:
                next_state = self.game.succ(state, act)
                a = max(a, self.alphabeta(next_state, depth-1, a, b, 1))
                if a >= b:
                    break
            return a

        ### Player 1 wants the point to be as small as possible(Min player)
        else:
            top_actions = self.get_top_actions(state, 1)
            for act in top_actions:
                next_state = self.game.succ(state, act)
                b = min(b, self.alphabeta(next_state, depth-1, a, b, 2))
                if a >= b:
                    break
            return b

    def getAction(self, state):
        """Get the next best action, including the first layer of minimax algorithm.
        Args:
            state: The current state of the board.
        """
        player = self.game.player(state)
        ### Player 1
        if player == 1:
            opt_val = float("inf")
            opt_action = []
            top_actions = self.get_top_actions(state, 1)
            for act in top_actions:
                next_state = self.game.succ(state, act)
                value = self.alphabeta(next_state, 1, float("-inf"), opt_val, 2)
                ### Record the optimal action list
                if value < opt_val:
                    opt_val = value
                    opt_action = [act]
                elif value == opt_val:
                    opt_action.append(act)
            self.action = random.choice(opt_action)
        ### Player 2
        else:
            opt_val = float("-inf")
            opt_action = []
            top_actions = self.get_top_actions(state, 2)
            for act in top_actions:
                next_state = self.game.succ(state, act)
                value = self.alphabeta(next_state, 1, opt_val, float("inf"), 1)
                ### Record the optimal action list
                if value > opt_val:
                    opt_val = value
                    opt_action = [act]
                elif value == opt_val:
                    opt_action.append(act)
            self.action = random.choice(opt_action)


    def oppAction(self, state):
        """Get the next best action for the opponent, including the first layer of minimax algorithm.
        Args:
            state: The current state of the board.
        """
        player = self.game.player(state)
        ### Player 1
        if player == 1:
            opt_val = float("-inf")
            opt_action = []
            top_actions = self.get_top_actions(state, 1)
            for act in top_actions:
                next_state = self.game.succ(state, act)
                value = self.alphabeta(next_state, 1, opt_val, float("inf"), 2)
                if value > opt_val:
                    opt_val = value
                    opt_action = [act]
                elif value == opt_val:
                    opt_action.append(act)
            self.opp_action = random.choice(opt_action)
        ### Player 2
        else:
            opt_val = float("inf")
            opt_action = []
            top_actions = self.get_top_actions(state, 2)
            for act in top_actions:
                next_state = self.game.succ(state, act)
                value = self.alphabeta(next_state, 1, float("-inf"), opt_val, 1)
                if value < opt_val:
                    opt_val = value
                    opt_action = [act]
                elif value == opt_val:
                    opt_action.append(act)
            self.opp_action = random.choice(opt_action)