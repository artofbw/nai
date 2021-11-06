"""
To jest gra w Connect 4
Autorzy:
Maciej Rybacki
Łukasz Ćwikliński
"""

from typing import List

from easyAI import TwoPlayerGame, Human_Player, Negamax, AI_Player


class ConnectFour(TwoPlayerGame):
    """
    Board game (6x7) to connect four dots in a row, column or diagonally.
    http://en.wikipedia.org/wiki/Connect_Four
    """
    ROW = 6
    COLUMN = 7
    WINNING_SEQUENCE = 4

    def __init__(self, players) -> None:
        """
        Init ConnectFour class with

        :param players: List
        :return: None
        """
        self.players = players
        self.board = [[0 for _ in range(self.COLUMN)] for _ in range(self.ROW)]
        self.current_player = 1

    def possible_moves(self) -> List:
        """
        Define possible moves
        :return: List
        """
        possible_moves = []

        for row in self.board:
            for index, col in enumerate(row):
                if col == 0:
                    possible_moves.append(index)

        return list(set(possible_moves))

    def make_move(self, column):
        column = int(column)

        for row in reversed(self.board):
            if row[column] == 0:
                row[column] = self.current_player
                break

    def show(self):
        """
        Display current board
        """
        for row in self.board:
            print(row)

    def is_over(self):
        """
        The game is over when there's no column to be played or there's a winning sequence
        """
        return all([min(row) != 0 for row in self.board]) or self.lose()

    def lose(self):
        """
        Lose if opponent have connected 4
        :return:
        """
        return self.connected_four(self.board, self.opponent_index)

    def scoring(self):
        return -100 if self.lose() else 0 # For the AI

    def connected_four(self, board: List, player: int) -> bool:
        """
        Returns True if player has connected four
        """
        # Check horizontal sequence
        for col in range(self.COLUMN - (self.WINNING_SEQUENCE - 1)):
            for row in range(self.ROW):
                if (
                        board[row][col:col + self.WINNING_SEQUENCE] ==
                        [player for _ in range(self.WINNING_SEQUENCE)]
                ):
                    return True

        # Check vertical sequence
        for col in range(self.COLUMN):
            for row in range(self.ROW - (self.WINNING_SEQUENCE - 1)):
                selected_board = board[row: row + self.WINNING_SEQUENCE]
                if [row[col] for row in selected_board] == [player for _ in range(self.WINNING_SEQUENCE)]:
                    return True

        # Check positive diaganol sequence
        for col in range(self.COLUMN - (self.WINNING_SEQUENCE - 1)):
            for row in range(self.ROW - (self.WINNING_SEQUENCE - 1)):
                if (
                        board[row][col] == player and
                        board[row + 1][col + 1] == player and
                        board[row + 2][col + 2] == player and
                        board[row + 3][col + 3] == player
                ):
                    return True

        # Check negative diaganol sequence
        for col in range(self.COLUMN - (self.WINNING_SEQUENCE - 1)):
            for row in range((self.WINNING_SEQUENCE - 1), self.ROW):
                if (
                        board[row][col] == player and
                        board[row - 1][col + 1] == player and
                        board[row - 2][col + 2] == player and
                        board[row - 3][col + 3] == player
                ):
                    return True

        return False


ai_algo_neg = Negamax(6)
connected_four = ConnectFour([Human_Player(), AI_Player(ai_algo_neg)])
history = connected_four.play()

if connected_four.lose():
    print(f"\nPlayer {connected_four.opponent_index} win")
