# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
import random
##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

class SarsaAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, epsilon=1, alpha=1, gamma=1, num_space=48):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_space = num_space
        self.Q_list = [{} for _ in range(self.num_space)]

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        ### Randomly choose at epsilon probability
        action = np.random.choice(self.all_actions)
        if random.random() < self.epsilon:
            return action
        ### Choose the action with the highest Q value
        max_q = float('-inf')
        for act, q in self.Q_list[observation].items():
            if q > max_q:
                action, max_q = act, q
        return action
    
    def learn(self, current_state, next_state, current_action, next_action, reward):
        """learn from experience"""
        self.Q_list[current_state][current_action] = (1 - self.alpha) * self.Q_list[current_state].get(current_action, 0) + self.alpha * (reward + self.gamma * self.Q_list[next_state].get(next_action, 0))
        return True

    ##### END CODING HERE #####


class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, epsilon=1, alpha=1, gamma=0.9, num_space=48):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_space = num_space
        self.Q_list = [{} for _ in range(self.num_space)]

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        ### Randomly choose at epsilon probability
        action = np.random.choice(self.all_actions)
        if random.random() < self.epsilon:
            return action
        ### Choose the action with the highest Q value
        max_q = float('-inf')
        for act, q in self.Q_list[observation].items():
            if q > max_q:
                action, max_q = act, q
        return action
    
    def learn(self, current_state, next_state, current_action, reward):
        """learn from experience"""
        max_q = float('-inf') if self.Q_list[next_state] else 0
        for _, q in self.Q_list[next_state].items():
            max_q = max(max_q, q)
        self.Q_list[current_state][current_action] = (1 - self.alpha) * self.Q_list[current_state].get(current_action, 0) + self.alpha * (reward + self.gamma * max_q)
        return True

    ##### END CODING HERE #####
    
    

class Dyna_QAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, epsilon=1, alpha=1, gamma=0.9, num_space=48, planning_steps=5):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_space = num_space
        self.Q_list = [{} for _ in range(self.num_space)]
        self.planning_steps = planning_steps
        self.model = {}

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        ### Randomly choose at epsilon probability
        action = np.random.choice(self.all_actions)
        if random.random() < self.epsilon:
            return action
        ### Choose the action with the highest Q value
        max_q = float('-inf')
        for act, q in self.Q_list[observation].items():
            if q > max_q:
                action, max_q = act, q
        return action
    
    def learn(self, current_state, next_state, current_action, reward):
        """learn from experience"""
        max_q = float('-inf') if self.Q_list[next_state] else 0
        for _, q in self.Q_list[next_state].items():
            max_q = max(max_q, q)
        self.Q_list[current_state][current_action] = (1 - self.alpha) * self.Q_list[current_state].get(current_action, 0) + self.alpha * (reward + self.gamma * max_q)

        self.model[(current_state, current_action)] = (next_state, reward)

        for _ in range(self.planning_steps):
            simulated_state, simulated_action = random.choice(list(self.model.keys()))
            simulated_next_state, simulated_reward = self.model[(simulated_state, simulated_action)]

            max_q = float('-inf') if self.Q_list[simulated_next_state] else 0
            for _, q in self.Q_list[simulated_next_state].items():
                max_q = max(max_q, q)
            self.Q_list[simulated_state][simulated_action] = (1 - self.alpha) * self.Q_list[simulated_state].get(simulated_action, 0) + self.alpha * (simulated_reward + self.gamma * max_q)
        return True

    ##### END CODING HERE #####
