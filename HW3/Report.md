# Report of HW3
Huanyu Wang 522030910212

## 1 Reinforcement Learning in Cliff-walking Environment

### 1.1 Agent Implementation

The agents are based on Sarsa, Q-Learning, and dyna-Q algorithms. The key components of the agent include:

- **\_\_init\_\_**: Initialize the agent using epsilon, alpha, gamma, etc.
    ```python
    def __init__(self, all_actions, epsilon=1, alpha=1, gamma=0.9, num_space=48):
    ```
- **choose_action**: The epsilon-greedy strategy to choose action.
    ```python
    def choose_action(self, observation):
    ```
- **learn**: The learning function, to update the Q-table.
    ```python
    def learn(self, current_state, next_state, current_action, reward):
    ```

### 1.2 Training Process

The agent is trained over multiple episodes. In each episode, the agent interacts with the environment, updates its Q-table based on the rewards received, and learns to avoid the cliff while reaching the goal. The training process involves:

1. For each episode:
    - Resetting the environment to the start state.
    - For each step in the episode:
      - Choosing an action using an ε-greedy policy.
      - Taking the action and observing the reward and next state.
      - Updating using learn(***) function.
      - Transitioning to the next state.
    - Reducing the exploration rate over time
        ```python
        agent.alpha *= 0.99
        agent.epsilon *= 0.99
        ```
        Exploration is encouraged to get familiar with the environment. And with the advancement of training, exploration is reduced for the convergence of the policy.
        ![epsilon](./pic/epsilon.png "epsilon")

**Notice**: the training processes of different agents are not the same, please view the corresponding cliff_walking_***.py

### 1.3 Results

- **Dyna-Q**

![dynaq-loss](./pic/dynaq-loss.png "dynaq-loss")

![dynaq-path](./pic/dynaq-path.png "dynaq-path")

The final episode reward is **-13**, which means that the agent chooses the optimal path toward the destination.

---

- **Q-learning**

![qlearning-loss](./pic/qlearning-loss.png "qlearning-loss")

![qlearning-path](./pic/qlearning-path.png "qlearning-path")

The final episode reward is also **-13**, which means that the agent chooses the optimal path toward the destination.

---

- **Sarsa**

![sarsa-loss](./pic/sarsa-loss.png "sarsa-loss")

![sarsa-path](./pic/sarsa-path.png "sarsa-path")

The final episode reward is **-17**, which means that the agent chooses the path that is not the optimal, but it is able to get to the destination eventually.

### 1.4 Analysis

- **Different paths**: The difference in the paths found by Sarsa and Q-learning can be attributed to the way they update their Q-values.
    - Sarsa is an **on-policy** algorithm, which means it updates its Q-values based on the action actually taken by the agent. This leads to a more **conservative** approach, as it takes into account the current policy's behavior.
    - Q-learning is an **off-policy** algorithm, which updates its Q-values based on the maximum possible reward for the next state, regardless of the current policy. This results in a more **aggressive** approach, as it always aims for the highest possible reward.

Thus, Sarsa chooses the path that is farther away from the cliff while Q-learning chooses the path that is next to the cliff. The latter path is optimal but more "dangerous", which is the result of an aggressive strategy.

- **The training efficiency**: As we know that the eventual episode rewards are -13 and -17 correspondingly, the number of episode the algorithm takes to reach this reward can evaluate the training efficiency:
    - Q-learning: **139**
    - Dyna-Q: **121**
    - Sarsa: **130**

As is shown above, (all the other parameters are the same) the model-based RL (dyna-Q) is **more efficient** than the model-free alorithms (Sarsa and Q-learning). The reason is that the model allows the dyna-Q agent to simulate experiences and plan actions based on simulations while other agents solely rely on direct interaction with the environment. As a result, the dyna-Q agent can receive more information every time it learns by simulating, thus more efficient when training.

## 2 DQN Implementation and Hyperparameters

The `dqn.py` implements the Deep Q-Network (DQN) algorithm for the cliff-walking environment where all the specific comments are filled in. The key components of the DQN agent include:

- **Neural Network**: The agent uses a neural network to approximate the Q-value function. The network takes the state as input and outputs Q-values for all possible actions.
    ```python
    class DQN(nn.Module):
    ```

- **Experience Replay**: The agent stores its experiences in a replay buffer and samples mini-batches from it to update the network.
    ```python
    class ReplayBuffer:
    ```

- **Training Loop**: The agent interacts with the environment, stores experiences, samples mini-batches, and updates the network using the Bellman equation.
    ```python
    for episode in range(num_episodes):
    ```

### 2.1 Hyperparameters

The hyperparameters used in the `dqn.py` script are as follows:

- **seed**: 42
- **total-timesteps**: 500000
- **learning-rate**: 1e-3
- **"buffer-size**: 1000
- **gamma**: 0.99
- **target-network-frequency**: 100
- **batch-size**: 128
- **start-e**: 0.3
- **end-e**: 0.05
- **exploration-fraction**: 0.1
- **learning-starts**: 10000
- **train-frequency**: 10

### 2.2 Analysis

- **Q-value**
![dqn-qvalue](./pic/dqn-qvalue.png "dqn-qvalue")

- **td-loss**
![dqn-tdloss](./pic/dqn-tdloss.png "dqn-tdloss")

After 500000 episodes of training, the aircraft can land very smoothly and accurately, which indicates that the DQN network is working very well. 
But, the td-loss curve seems very complex. The loss is quite low at roughly 150k episode. However, when I tried to only train for 150k episodes, the result is not as perfect as the original one, which indicates that the DQN network is still learning even though the loss is low(might be some kind of local optimum).

- **Tuning the hyper-parameters**: I've tried increasing the batch-size, increasing the target-network-frequency, etc, all of which is worth than the current parameters, which is hard to tune.
- **Tuning the structure of the Q network**: I also tried adding one more layer to the Q network:
    ```python
    self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 100),
            nn.ReLU(),
            nn.Linear(100, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )
    ```
    But the result is a little bit worth, even after I increased the learning rate, thinking that deeper network may require a larger learning rate. I find that this deeper Q network makes the aircraft landing more slowly, seems more "conservative".

- **Concluesion**: The hyper-parameters listed above perform best, which is difficult to outperform.

## 3 Improve Exploration Schema

### 3.1 Upper Confidence Bound (UCB) Exploration

Upper Confidence Bound (UCB) is an exploration strategy that balances exploration and exploitation by considering both the average reward and the uncertainty of actions. The key idea is to select actions based on their potential to yield high rewards while also accounting for the uncertainty in the estimates.

#### Idea

The UCB algorithm selects actions by maximizing the following equation:
\[ a_t = \arg\max_{a} \left( Q(a) + c \sqrt{\frac{\ln(t)}{N(a)}} \right) \]
where:
- \( Q(a) \) is the estimated reward for action \( a \).
- \( c \) is a constant that controls the degree of exploration.
- \( t \) is the total number of times any action has been selected.
- \( N(a) \) is the number of times action \( a \) has been selected.

The first term \( Q(a) \) promotes exploitation by favoring actions with high estimated rewards. The second term \( c \sqrt{\frac{\ln(t)}{N(a)}} \) promotes exploration by favoring actions that have been selected less frequently, thus having higher uncertainty.

#### Pros

- **Balanced Exploration and Exploitation**: UCB provides a principled way to balance exploration and exploitation, ensuring that actions with high uncertainty are explored sufficiently.
- **Theoretical Guarantees**: UCB has strong theoretical guarantees and is proven to achieve logarithmic regret in multi-armed bandit problems.
- **Adaptability**: The algorithm adapts to changes in the environment by continuously updating the action selection based on new observations.

#### Cons

- **Computational Complexity**: UCB requires maintaining and updating counts for each action, which can be computationally expensive in environments with a large action space.
- **Parameter Sensitivity**: The performance of UCB is sensitive to the choice of the exploration constant \( c \). An inappropriate value can lead to suboptimal performance.
- **Delayed Exploitation**: In some cases, UCB may delay exploitation of high-reward actions due to the exploration term, especially in the early stages of learning.

Overall, UCB is a powerful exploration strategy that can be particularly useful in environments where the balance between exploration and exploitation is crucial for optimal performance.