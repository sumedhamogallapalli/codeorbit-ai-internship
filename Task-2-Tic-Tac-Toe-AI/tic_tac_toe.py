"""
CodeOrbit Tech - Artificial Intelligence Internship
Task 2: Tic-Tac-Toe with Simple AI (Minimax Algorithm)
Author: Internship Candidate (Final-Year B.Tech Student)
Technology: Python 3 (Standard Library Only)
"""

import math

# Board positions are represented by a list of 9 elements (indices 0-8)
# Corresponding to positions 1-9 on a numeric keypad / board:
#  1 | 2 | 3
# ---+---+---
#  4 | 5 | 6
# ---+---+---
#  7 | 8 | 9

HUMAN = 'X'
AI = 'O'
EMPTY = ' '

def print_board(board):
    """Prints the current state of the 3x3 game board."""
    print()
    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("---+---+---")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("---+---+---")
    print(f" {board[6]} | {board[7]} | {board[8]} ")
    print()

def print_instructions():
    """Displays board position guide for the player."""
    print("\n--- WELCOME TO TIC-TAC-TOE WITH AI ---")
    print("You are 'X' and Computer (AI) is 'O'.")
    print("Positions are numbered 1 through 9 as shown below:")
    print(" 1 | 2 | 3 ")
    print("---+---+---")
    print(" 4 | 5 | 6 ")
    print("---+---+---")
    print(" 7 | 8 | 9 ")
    print("--------------------------------------")

def check_winner(board, player):
    """
    Checks if the specified player has won the game.
    Returns True if the player has 3 in a row, column, or diagonal.
    """
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] == player:
            return True
    return False

def is_board_full(board):
    """Returns True if there are no empty spaces left on the board."""
    return EMPTY not in board

def get_empty_positions(board):
    """Returns a list of indices (0-8) that are currently empty."""
    return [i for i, spot in enumerate(board) if spot == EMPTY]

def minimax(board, depth, is_maximizing):
    """
    Minimax Algorithm:
    - AI is the maximizing player (aims for +10 score).
    - Human is the minimizing player (aims for -10 score).
    - Depth penalizes longer paths so the AI wins in fewer moves.
    """
    if check_winner(board, AI):
        return 10 - depth
    if check_winner(board, HUMAN):
        return depth - 10
    if is_board_full(board):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for pos in get_empty_positions(board):
            board[pos] = AI
            score = minimax(board, depth + 1, False)
            board[pos] = EMPTY
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for pos in get_empty_positions(board):
            board[pos] = HUMAN
            score = minimax(board, depth + 1, True)
            board[pos] = EMPTY
            best_score = min(score, best_score)
        return best_score

def get_best_move(board):
    """
    Evaluates all possible moves for the AI using Minimax
    and returns the best position index (0-8).
    """
    best_score = -math.inf
    best_move = None

    for pos in get_empty_positions(board):
        board[pos] = AI
        score = minimax(board, 0, False)
        board[pos] = EMPTY

        if score > best_score:
            best_score = score
            best_move = pos

    return best_move

def human_move(board):
    """Prompts human player to enter a move (1-9) and validates input."""
    while True:
        try:
            choice = input("Enter your move (1-9): ").strip()
            if not choice.isdigit():
                print("Invalid input! Please enter a number between 1 and 9.")
                continue

            pos = int(choice) - 1
            if pos < 0 or pos > 8:
                print("Invalid position! Please choose a number from 1 to 9.")
            elif board[pos] != EMPTY:
                print("That spot is already taken! Choose another spot.")
            else:
                board[pos] = HUMAN
                break
        except (ValueError, KeyboardInterrupt, EOFError):
            print("\nGame interrupted. Exiting.")
            exit(0)

def play_game():
    """Main game loop for single match."""
    board = [EMPTY] * 9
    print_instructions()
    print_board(board)

    while True:
        # 1. Human Turn
        print("Your Turn (X):")
        human_move(board)
        print_board(board)

        if check_winner(board, HUMAN):
            print("Congratulations! You won the game!")
            break
        if is_board_full(board):
            print("It's a draw!")
            break

        # 2. AI Turn
        print("Computer's Turn (O)... Thinking...")
        ai_pos = get_best_move(board)
        board[ai_pos] = AI
        print(f"Computer placed 'O' at position {ai_pos + 1}.")
        print_board(board)

        if check_winner(board, AI):
            print("Computer (AI) wins! Better luck next time.")
            break
        if is_board_full(board):
            print("It's a draw!")
            break

def main():
    """Program entry point with replay option."""
    while True:
        play_game()
        try:
            choice = input("\nDo you want to play again? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if choice != 'y':
            print("Thank you for playing Tic-Tac-Toe! Goodbye.")
            break

if __name__ == "__main__":
    main()
