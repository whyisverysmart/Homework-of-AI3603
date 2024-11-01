# -*- coding:utf-8 -*-
# Train Q-Learning in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import QLearningAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
import matplotlib.pyplot as plt
##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0")
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 

##### START CODING HERE #####

# construct the intelligent agent.
agent = QLearningAgent(all_actions,
                        epsilon=1,
                        alpha=1,
                        gamma=0.9,
                        num_space=48)

# For episode_reward visualization
reward_list = []

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    current_state = env.reset()
    # render env. You can remove all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # choose an action
        current_action = agent.choose_action(current_state)
        next_state, reward, isdone, info = env.step(current_action)
        # env.render()
        # update the episode reward
        episode_reward += reward
        # print(f"{current_state} {action} {next_state} {reward} {isdone}")
        # agent learns from experience
        agent.learn(current_state, next_state, current_action, reward)
        current_state = next_state
        if isdone:
            agent.Q_list[current_state][current_action] = reward
            break
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)
    reward_list.append(episode_reward)
    # agent.alpha -= 0.001
    # agent.epsilon -= 0.001
    agent.alpha *= 0.99
    agent.epsilon *= 0.99

print('\ntraining over\n')

# Episode_reward visualization
plt.plot(range(1000), reward_list)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Episode Reward over Time')
plt.show()

# Final path visualization
agent.epsilon = 0
current_state = env.reset()
env.render()
while True:
    current_action = agent.choose_action(current_state)
    next_state, reward, isdone, info = env.step(current_action)
    time.sleep(0.5)
    env.render()
    current_state = next_state
    if isdone:
        break

# close the render window after training.
env.close()

##### END CODING HERE #####


