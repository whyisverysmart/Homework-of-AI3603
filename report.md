### Report: Minimax Algorithm with Alpha-Beta Pruning in New Checkers

---

#### 1. **Objective**

The task is to implement an AI agent (`YourAgent`) using the **Minimax algorithm with Alpha-Beta Pruning** to play the game of the new Checkers. The performance of this agent is tested against the `SimpleGreedyAgent`, which uses a simple greedy strategy. Additionally, a match between two instances of `YourAgent` was conducted.

P.S. **Movement Rules**
Each turn, a player can move one piece. Pieces can move in two ways, following standard Chinese Checkers rules:
- Move to an adjacent empty space: A piece can move to any of the directly adjacent positions if they are unoccupied.
- Jump over a piece: A piece can jump over an adjacent piece (belonging to either player) if the space directly on the opposite side, along the same line, is empty. Multiple jumps are allowed in a single move if conditions allow.

---

#### 2. **Implementation of Minimax Algorithm with Alpha-Beta Pruning**

The `YourAgent` class was completed to implement the Minimax algorithm with Alpha-Beta pruning. This search algorithm is a depth-first recursive procedure used to minimize the possible loss for a worst-case scenario (minimizing the opponent's gain), while alpha-beta pruning optimizes the search by pruning branches that cannot improve the outcome, thus reducing the computational cost.

Key components implemented in `YourAgent`:

- **Minimax with Alpha-Beta Pruning:** The core function recursively explores all possible moves up to a specified depth.
    ```python
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
    ```

- **Evaluation Function:** A heuristic function to evaluate the board state, which returns a total point.
    ```python
    def eval_state(self, state):
    ```
    - For **Normal Pieces:**
    ```python
    elif value == 1:
        ### 1. Sum the row of piece
        ### 2. Bonus and Penalty:
        ###    - Should maintain in center
        ###    - Should not occupy the special position
        row1 += key[0]
        total_point += abs(key[1] - min(key[0]+1, 21-key[0])/2) * center_bonus_ratio
        if key[0] == 2:
            total_point += 100
    ```
    ```python
    total_point += (row1 + row2) * normal_bonus_ratio
    ```
    - For **Special Pieces:**
    ```python
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
    ```
    - Check the winning condition
    ```python
    ### Winning condition
    if row1 == 30 and (status[(2, 1)] == 3 and status[(2, 2)] == 3):
        return -100000
    if row2 == 170 and (status[(18, 1)] == 4 and status[(18, 2)] == 4):
        return 100000
    ```

- **Get top actions Function:** To get the top 1/2 actions with max evaluation.
```python
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
```

---

#### 3. **Testing Setup**

The following tests were conducted:

- **Test 1:** `YourAgent` (Player 2) vs. `SimpleGreedyAgent` (Player 1)
- **Test 2:** `YourAgent` (Player 1) vs. `SimpleGreedyAgent` (Player 2)
- **Test 3:** `YourAgent` (Player 1) vs. `YourAgent` (Player 2)

Each game was run using the `runGame.py`, with a fixed depth limit of 2 for the minimax search to balance performance and accuracy.

---

#### 4. **Results**

**Test 1: YourAgent (Player 2) vs. SimpleGreedyAgent (Player 1)**

- **Result:** `YourAgent` wins after **78.7** moves in average.
```
winner is 2 in 78
winner is 2 in 90
winner is 2 in 68
winner is 2 in 68
winner is 2 in 88
winner is 2 in 84
winner is 2 in 80
winner is 2 in 96
winner is 2 in 80...
```

**Test 2: YourAgent (Player 1) vs. SimpleGreedyAgent (Player 2)**

- **Result:** `YourAgent` wins after **81.8** moves in average.
```
winner is 1 in 77
winner is 1 in 87
winner is 1 in 83
winner is 1 in 73
winner is 1 in 77
winner is 1 in 101
winner is 1 in 83
winner is 1 in 79
winner is 1 in 77...
```
- **Observations:**
  - In some special cases, the game may stuck, which is because one of the piece may move slower, and the agent considers it not optimal to move it when the opponent is waiting for its position.
  - Because tne agent will randomly choose the action from a list of optimal actions, it may be unstable, but it can avoid some stucking situation.
  - YourAgent is much stabler when acting as player 1.

**Test 3: YourAgent (Player 1) vs. YourAgent (Player 2)**

- **Result:** Player 1 wins 6 games while player 2 wins 1 games. And 3 ties.
```
winner is 1 in 131
stuck
winner is 2 in 140
winner is 1 in 131
winner is 1 in 87...
```
- **Observations:**
  - They compete fiercely, and their capability is close to each other.
  - Sometimes the stuck situation still happens because the agent considers it not optimal to move the piece when it occupies one position of the opponent, which makes the opponent unable to win.

---

#### 5. **Analysis and Conclusions**

The Minimax algorithm with Alpha-Beta pruning allowed `YourAgent` to play a significantly more strategic game compared to the `SimpleGreedyAgent`. The key strengths of `YourAgent` included:

- **Strategic Depth:** The ability to evaluate positions more deeply and avoid short-term losses for long-term gains.
- **Efficiency:** Alpha-Beta pruning drastically reduced the number of nodes explored during the search, allowing for deeper analysis within the same computational limits.

However, Test 3 showed that while `YourAgent` performs well against simpler strategies, additional improvements in the evaluation function could lead to more decisive victories against similarly advanced agents. Future enhancements could include more sophisticated heuristics or adaptive depth search.

---

#### 6. **Future Improvements**

Potential improvements to the `YourAgent` implementation include:
- **Advanced Heuristic Tuning:** Adding more factors that could improve decision-making.
    - There are many **magic numbers** in the evaluation function, which can be further optimized!
    - More bonus and penalty can be added to restrict the agent to perform better! But I've tried many other bonuses, including encouraging the last piece to move faster, encouraging the piece to get into inside first...These only made the evaluation function uglier, hardly enhancing the performance.
- **Adaptive Depth Search:** Using iterative deepening or dynamic depth adjustment based on game stage could enhance both efficiency and effectiveness.

---