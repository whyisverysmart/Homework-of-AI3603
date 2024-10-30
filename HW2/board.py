class Board(object):
    """
    Board class represents a game board for a two-player game.

    Attributes:
        size (int): The size of the board.
        piece_rows (int): The number of rows occupied by pieces at the start.
        max_iter (int): The maximum number of iterations allowed.
        player1_pos (dict): The initial positions of player 1's pieces.
        player2_pos (dict): The initial positions of player 2's pieces.
        board_status (dict): The status of each position on the board.

    Methods:
        __init__(size, piece_rows, max_iter=200):
            Initializes the board with the given size, piece rows, and maximum iterations.

        getColNum(row):
            Returns the number of columns in the given row.

        isEmptyPosition(pos):
            Checks if the given position is empty.

        leftPosition(pos):
            Returns the position to the left of the given position.

        rightPosition(pos):
            Returns the position to the right of the given position.

        upLeftPosition(pos):
            Returns the position to the upper left of the given position.

        upRightPosition(pos):
            Returns the position to the upper right of the given position.

        downLeftPosition(pos):
            Returns the position to the lower left of the given position.

        downRightPosition(pos):
            Returns the position to the lower right of the given position.

        adjacentPositions(pos):
            Returns a list of positions adjacent to the given position.

        getPlayerPiecePositions(player):
            Returns a list of positions occupied by the given player's pieces.

        getOneDirectionHopPosition(pos, dir_func):
            Returns the possible target hop position in the direction designated by dir_func.

        getOneHopPositions(pos):
            Returns a list of positions that can be reached from the given position in one hop.

        getAllHopPositions(pos):
            Returns all positions that can be reached from the given position in several hops.

        compare_piece_num():
            Compares the number of pieces for both players and returns the player with more pieces.

        ifPlayerWin(player, iter):
            Checks if the given player has won the game.

        isEnd(iter):
            Checks if the game has ended and returns the result.

        printBoard():
            Prints the current state of the board.

        printBoardOriginal():
            Prints the original state of the board.
    """

    def __init__(self, size, piece_rows, max_iter=200):
        assert piece_rows < size

        self.player1_pos = {'(2, 1)': False, '(2, 2)': False}
        self.player2_pos = {'(18, 1)': False, '(18, 2)': False}

        self.size = size
        self.piece_rows = piece_rows
        self.max_iter = max_iter
        self.board_status = {}
        for row in range(1, size + 1):
            for col in range(1, self.getColNum(row) + 1):
                if row <= piece_rows:
                    self.board_status[(row, col)] = 2
                else:
                    self.board_status[(row, col)] = 0

        self.board_status[(2, 1)] = 4
        self.board_status[(2, 2)] = 4

        for row in range(size + 1, size * 2):
            for col in range(1, self.getColNum(row) + 1):
                if row < size * 2 - piece_rows:
                    self.board_status[(row, col)] = 0
                else:
                    self.board_status[(row, col)] = 1

        self.board_status[(size * 2 - piece_rows + 2, 1)] = 3
        self.board_status[(size * 2 - piece_rows + 2, 2)] = 3

    def getColNum(self, row):
        if row in range(1, self.size + 1):
            return row
        else:
            return self.size * 2 - row

    def isEmptyPosition(self, pos):
        return self.board_status[pos] == 0

    def leftPosition(self, pos):
        row = pos[0]
        col = pos[1]
        if (row, col - 1) in list(self.board_status.keys()):
            return (row, col - 1)

    def rightPosition(self, pos):
        row = pos[0]
        col = pos[1]
        if (row, col + 1) in list(self.board_status.keys()):
            return (row, col + 1)

    def upLeftPosition(self, pos):
        row = pos[0]
        col = pos[1]
        if row <= self.size and (row - 1, col - 1) in list(self.board_status.keys()):
            return (row - 1, col - 1)
        if row > self.size and (row - 1, col) in list(self.board_status.keys()):
            return (row - 1, col)

    def upRightPosition(self, pos):
        row = pos[0]
        col = pos[1]
        if row <= self.size and (row - 1, col) in list(self.board_status.keys()):
            return (row - 1, col)
        if row > self.size and (row - 1, col + 1) in list(self.board_status.keys()):
            return (row - 1, col + 1)

    def downLeftPosition(self, pos):
        row = pos[0]
        col = pos[1]
        if row < self.size and (row + 1, col) in list(self.board_status.keys()):
            return (row + 1, col)
        if row >= self.size and (row + 1, col - 1) in list(self.board_status.keys()):
            return (row + 1, col - 1)

    def downRightPosition(self, pos):
        row = pos[0]
        col = pos[1]
        if row < self.size and (row + 1, col + 1) in list(self.board_status.keys()):
            return (row + 1, col + 1)
        if row >= self.size and (row + 1, col) in list(self.board_status.keys()):
            return (row + 1, col)

    def adjacentPositions(self, pos):
        result = []
        result.append(self.leftPosition(pos))
        result.append(self.rightPosition(pos))
        result.append(self.upLeftPosition(pos))
        result.append(self.upRightPosition(pos))
        result.append(self.downLeftPosition(pos))
        result.append(self.downRightPosition(pos))
        return [x for x in result if x is not None]

    def getPlayerPiecePositions(self, player):
        # return a list of positions that player's pieces occupy
        result1 = [
            (row, col)
            for row in range(1, self.size + 1)
            for col in range(1, self.getColNum(row) + 1)
            if self.board_status[(row, col)] == player or self.board_status[(row, col)] == player + 2
        ]
        result2 = [
            (row, col)
            for row in range(self.size + 1, self.size * 2)
            for col in range(1, self.getColNum(row) + 1)
            if self.board_status[(row, col)] == player or self.board_status[(row, col)] == player + 2
        ]
        return result1 + result2

    def getOneDirectionHopPosition(self, pos, dir_func):
        # return possible target hop position in the direction designated by dir_func
        # our rule: can hop as long as there's only one piece on the line between current position and target position
        # and the piece hopped over is at the middle point
        hop_over_pos = dir_func(pos)
        count = 0
        while hop_over_pos is not None:
            if self.board_status[hop_over_pos] != 0:
                break
            hop_over_pos = dir_func(hop_over_pos)
            count += 1
        if hop_over_pos is not None:
            target_position = dir_func(hop_over_pos)
            while count > 0:
                if target_position is None or self.board_status[target_position] != 0:
                    break
                target_position = dir_func(target_position)
                count -= 1
            if count == 0 and target_position is not None and self.board_status[target_position] == 0:
                return target_position

    def getOneHopPositions(self, pos):
        result = []
        result.append(self.getOneDirectionHopPosition(pos, self.leftPosition))
        result.append(self.getOneDirectionHopPosition(pos, self.rightPosition))
        result.append(self.getOneDirectionHopPosition(pos, self.upLeftPosition))
        result.append(self.getOneDirectionHopPosition(pos, self.upRightPosition))
        result.append(self.getOneDirectionHopPosition(pos, self.downLeftPosition))
        result.append(self.getOneDirectionHopPosition(pos, self.downRightPosition))
        return [x for x in result if x is not None]

    def getAllHopPositions(self, pos):
        # return all positions can be reached from current position in several hops
        result = self.getOneHopPositions(pos)
        start_index = 0
        while start_index < len(result):
            cur_size = len(result)
            for i in range(start_index, cur_size):
                for new_pos in self.getOneHopPositions(result[i]):
                    if new_pos not in result:
                        result.append(new_pos)
            start_index = cur_size
            if pos in result:
                result.remove(pos)
        return result

    def compare_piece_num(self):

        player1_score = 0
        player2_score = 0

        for row in range(1, self.piece_rows + 1):
            for col in range(1, self.getColNum(row) + 1):
                if self.board_status[(row, col)] == 3 or self.board_status[(row, col)] == 1:
                    player1_score += 1

        for row in range(self.size * 2 - self.piece_rows, self.size * 2):
            for col in range(1, self.getColNum(row) + 1):
                if self.board_status[(row, col)] == 4 or self.board_status[(row, col)] == 2:
                    player2_score += 1

        if player1_score > player2_score:
            return 1
        else:
            return 0

    def ifPlayerWin(self, player, iter):
        if player == 1:
            for row in range(1, self.piece_rows + 1):
                for col in range(1, self.getColNum(row) + 1):
                    if row == 2 and self.board_status[(row, col)] == 3:
                        continue
                    elif self.board_status[(row, col)] == 1:
                        continue
                    elif iter > self.max_iter:
                        return self.compare_piece_num()
                    else:
                        return False
            return True

        else:
            for row in range(self.size * 2 - self.piece_rows, self.size * 2):
                for col in range(1, self.getColNum(row) + 1):
                    if row == self.size * 2 - self.piece_rows + 2 and self.board_status[(row, col)] == 4:
                        continue
                    elif self.board_status[(row, col)] == 2:
                        continue
                    elif iter > self.max_iter:
                        return not self.compare_piece_num()
                    else:
                        return False
            return True

    def isEnd(self, iter):
        player_1_reached = self.ifPlayerWin(1, iter)
        player_2_reached = self.ifPlayerWin(2, iter)
        print(f'iterations: {iter}')
        if player_1_reached:
            return (True, 1)
        if player_2_reached:
            return (True, 2)
        return (False, None)

    def printBoard(self):
        for row in range(1, self.size + 1):
            print(' ' * (self.size - row), end=' ')
            for col in range(1, self.getColNum(row) + 1):
                print(str(self.board_status[(row, col)]), end=' ')

            print('\n', end=' ')

        for row in range(self.size + 1, self.size * 2):
            print(' ' * (row - self.size), end=' ')
            for col in range(1, self.getColNum(row) + 1):
                print(str(self.board_status[(row, col)]), end=' ')

            print('\n', end=' ')

    def printBoardOriginal(self):
        for row in range(1, self.size + 1):
            for col in range(1, self.getColNum(row) + 1):
                print(str(self.board_status[(row, col)]), end=' ')

            print('\n', end=' ')

        for row in range(self.size + 1, self.size * 2):
            for col in range(1, self.getColNum(row) + 1):
                print(str(self.board_status[(row, col)]), end=' ')

            print('\n', end=' ')
