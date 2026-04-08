"""
Tic Tac Toe Player
"""

import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Count number of X's and O's on the board
    x_count = 0
    o_count = 0
    
    for row in board:
        for cell in row:
            if cell == X:
                x_count += 1
            elif cell == O:
                o_count += 1
    
    # X goes first, so if counts are equal, it's X's turn
    if x_count == o_count:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Check if action is valid
    if action not in actions(board):
        raise Exception("Invalid action")
    
    # Create a deep copy of the board
    new_board = copy.deepcopy(board)
    i, j = action
    
    # Make the move for the current player
    new_board[i][j] = player(board)
    
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not EMPTY:
            return board[i][0]
    
    # Check columns
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] and board[0][j] is not EMPTY:
            return board[0][j]
    
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not EMPTY:
        return board[0][0]
    
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not EMPTY:
        return board[0][2]
    
    # No winner
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Game is over if there's a winner
    if winner(board) is not None:
        return True
    
    # Game is over if all cells are filled
    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False
    
    # All cells filled and no winner -> tie
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # If board is terminal, return None
    if terminal(board):
        return None
    
    current_player = player(board)
    
    if current_player == X:
        # X wants to maximize the utility
        value, best_move = max_value(board)
        return best_move
    else:
        # O wants to minimize the utility
        value, best_move = min_value(board)
        return best_move


def max_value(board):
    """
    Returns the maximum utility value and the best move for the maximizing player.
    """
    # If board is terminal, return utility and no move
    if terminal(board):
        return utility(board), None
    
    v = -math.inf
    best_move = None
    
    for action in actions(board):
        # Get the minimum value from the opponent's perspective
        min_val, _ = min_value(result(board, action))
        
        # Update best move if we found a better value
        if min_val > v:
            v = min_val
            best_move = action
            
            # Pruning: if we found the maximum possible value, we can stop
            if v == 1:
                return v, best_move
    
    return v, best_move


def min_value(board):
    """
    Returns the minimum utility value and the best move for the minimizing player.
    """
    # If board is terminal, return utility and no move
    if terminal(board):
        return utility(board), None
    
    v = math.inf
    best_move = None
    
    for action in actions(board):
        # Get the maximum value from the opponent's perspective
        max_val, _ = max_value(result(board, action))
        
        # Update best move if we found a better value
        if max_val < v:
            v = max_val
            best_move = action
            
            # Pruning: if we found the minimum possible value, we can stop
            if v == -1:
                return v, best_move
    
    return v, best_move