# CodeOrbit Tech - Artificial Intelligence Internship
## Task 2: Tic-Tac-Toe with Simple AI (Minimax Algorithm)

A clean, console-based Tic-Tac-Toe game developed in pure Python (Standard Library only) featuring an intelligent AI opponent powered by the Minimax algorithm.

---

## 1. Project Folder Structure

```
tic-tac-toe-ai/
│
├── tic_tac_toe.py    # Main game script containing game logic and the Minimax AI algorithm
├── requirements.txt  # Project dependencies (Standard library only; zero third-party packages)
└── README.md         # Comprehensive documentation, execution steps, and interview guide
```

---

## 2. Installation & Setup Instructions

### Prerequisites
- Python 3.8 or higher installed on your computer.
- You can check your Python version by running:
  ```bash
  python3 --version
  ```
  *(or `python --version` on Windows)*

### Setup
1. Clone or download this project folder to your local machine.
2. Open your terminal or command prompt and navigate to the project directory:
   ```bash
   cd tic-tac-toe-ai
   ```
3. No external libraries are needed (`requirements.txt` requires no `pip install`).

---

## 3. How to Run the Game

Execute the script with Python:

```bash
python3 tic_tac_toe.py
```
*(On Windows command prompt/PowerShell: `python tic_tac_toe.py`)*

---

## 4. How to Play

- The game board is a 3x3 grid with positions numbered from **1 to 9**:
  ```
   1 | 2 | 3
  ---+---+---
   4 | 5 | 6
  ---+---+---
   7 | 8 | 9
  ```
- You play as **'X'** (first turn).
- The Computer (AI) plays as **'O'**.
- When prompted, enter a number from `1` to `9` corresponding to the cell you wish to mark.
- The game validates every input to prevent overwriting an already chosen spot or entering out-of-range values.
- After every turn, the board is updated and printed in the terminal.
- The match ends when someone gets 3 in a row/column/diagonal or when all 9 cells are filled (Draw).
- At game end, you are asked if you'd like to play again (`y/n`).

---

## 5. Sample Gameplay & Output

```
--- WELCOME TO TIC-TAC-TOE WITH AI ---
You are 'X' and Computer (AI) is 'O'.
Positions are numbered 1 through 9 as shown below:
 1 | 2 | 3 
---+---+---
 4 | 5 | 6 
---+---+---
 7 | 8 | 9 
--------------------------------------

   |   |   
---+---+---
   |   |   
---+---+---
   |   |   

Your Turn (X):
Enter your move (1-9): 5

   |   |   
---+---+---
   | X |   
---+---+---
   |   |   

Computer's Turn (O)... Thinking...
Computer placed 'O' at position 1.

 O |   |   
---+---+---
   | X |   
---+---+---
   |   |   

Your Turn (X):
Enter your move (1-9): 2

 O | X |   
---+---+---
   | X |   
---+---+---
   |   |   

Computer's Turn (O)... Thinking...
Computer placed 'O' at position 8.

 O | X |   
---+---+---
   | X |   
---+---+---
   | O |   

Your Turn (X):
Enter your move (1-9): 3

 O | X | X 
---+---+---
   | X |   
---+---+---
   | O |   

Computer's Turn (O)... Thinking...
Computer placed 'O' at position 7.

 O | X | X 
---+---+---
   | X |   
---+---+---
 O | O |   

Your Turn (X):
Enter your move (1-9): 9

 O | X | X 
---+---+---
   | X |   
---+---+---
 O | O | X 

Computer's Turn (O)... Thinking...
Computer placed 'O' at position 4.

 O | X | X 
---+---+---
 O | X |   
---+---+---
 O | O | X 

Computer (AI) wins! Better luck next time.

Do you want to play again? (y/n): n
Thank you for playing Tic-Tac-Toe! Goodbye.
```

---

## 6. Simple Explanation of How the AI Works (Minimax Algorithm)

The AI uses the **Minimax Algorithm**, which is a decision-making algorithm from Game Theory used in two-player, turn-based, zero-sum games.

### How it works in simple words:
1. **Zero-Sum Game:** One player's gain is the other player's loss.
2. **Two Roles:**
   - **Maximizer (AI):** Wants to maximize the outcome score (aims for `+10` for winning).
   - **Minimizer (Human):** Wants to minimize the AI's score (aims for `-10` for winning).
   - **Draw:** Results in a score of `0`.
3. **Simulating the Future:**
   - When it is the AI's turn, it simulates all available empty spots.
   - For every move, it recursively simulates every possible counter-move the human could make, all the way to terminal states (win, lose, or draw).
4. **Depth Consideration:**
   - The algorithm prefers quick wins over slow wins: `score = 10 - depth` for AI wins, and `score = depth - 10` for human wins.
5. **Backtracking the Best Move:**
   - The AI assumes the human will always play their best possible move (optimal play).
   - Working backwards up the game tree, the AI selects the move that guarantees the highest possible payoff regardless of what move the human chooses.
   - Because Tic-Tac-Toe is a solved game, the Minimax AI will **never lose**. It will either win or force a draw!

---

## 7. Explanation of Important Functions

1. `print_board(board)`:
   - Formats and displays the 9-element board list as an easy-to-read 3×3 grid with separators (`|` and `---`).

2. `check_winner(board, player)`:
   - Checks all 8 possible winning combinations (3 rows, 3 columns, 2 diagonals).
   - Returns `True` if the given player ('X' or 'O') occupies all three positions in any combination.

3. `is_board_full(board)`:
   - Checks if any `' '` (empty space) remains on the board. Used to detect a draw condition.

4. `get_empty_positions(board)`:
   - Finds and returns a list of indices (0 to 8) that are currently unoccupied.

5. `minimax(board, depth, is_maximizing)`:
   - The core recursive search function.
   - Evaluates terminal states: returns positive score if AI wins, negative if human wins, 0 if tie.
   - Recursively traverses all child branches to compute the optimal score for the current player's perspective.

6. `get_best_move(board)`:
   - Iterates through every currently available cell on the board.
   - Calls `minimax()` on each available spot.
   - Returns the index corresponding to the move that achieved the maximum evaluation score.

7. `human_move(board)`:
   - Takes input from the player via `input()`.
   - Validates that the input is a valid number between 1 and 9, and ensures the chosen cell is not already taken.

8. `play_game()`:
   - Manages the primary match flow: initializes board, alternates between human and AI turns, displays the board, checks for win/draw after each move, and declares the result.

9. `main()`:
   - Wraps the game in a replay loop, allowing the user to start a new match without restarting the script.

---

## 8. 2 to 5-Minute Screen Recording Demo Script

| Timestamp | Screen Action | Spoken Script / Narration |
| :--- | :--- | :--- |
| **0:00 - 0:30** | Show VS Code / Terminal with file tree (`tic_tac_toe.py`, `requirements.txt`, `README.md`). Show `python3 --version`. | *"Hello everyone! My name is [Your Name], and this is Task 2 of my Artificial Intelligence Internship at CodeOrbit Tech: Building a Tic-Tac-Toe Game with an AI Opponent using Python."* |
| **0:30 - 1:15** | Run `python3 tic_tac_toe.py`. Enter invalid moves (`abc`, `10`, or occupied cells). | *"Let's run the game using `python3 tic_tac_toe.py`. First, notice the clean board with position guides from 1 to 9. If I enter an invalid input like 'abc' or '10', the game cleanly catches it with helpful feedback and doesn't crash."* |
| **1:15 - 2:15** | Play a round demonstrating the AI blocking the player and winning. | *"Now let's play. I place 'X' in position 1. The computer analyzes the board and takes the center position 5. Next, I play position 2. The AI immediately identifies that I have two in a row and blocks me at position 3! As the game continues, the AI executes a winning move and announces Computer Wins."* |
| **2:15 - 3:00** | Play a second round (enter `y`) playing defensively to demonstrate an optimal Draw match. | *"Let's test the replay feature by pressing 'y'. This time, I will play an optimal strategy. Notice how every countermove by the AI is calculated via the Minimax algorithm. The game reaches all 9 moves and cleanly declares a Draw. Because Minimax explores the entire game tree, the AI can never be defeated."* |
| **3:00 - 3:45** | Briefly show `tic_tac_toe.py` focusing on `minimax()` and `check_winner()`. | *"Taking a quick look at the code: we utilized pure standard Python without any external dependencies. The `minimax` function recursively evaluates all possible game states, assigning positive scores for AI wins and negative for human wins."* |
| **3:45 - 4:15** | Conclude the video, exit game (`n`), thank the audience and CodeOrbit Tech. | *"This completes Task 2 for CodeOrbit Tech. Thank you for watching!"* |

---

## 9. LinkedIn Post Template

```text
🚀 Excited to share my Task 2 submission for the Artificial Intelligence Internship at CodeOrbit Tech!

For this task, I built a console-based Tic-Tac-Toe game in Python featuring an intelligent AI opponent powered by the Minimax algorithm. 🎮🤖

Key Highlights:
✨ Pure Python 3 implementation utilizing only the standard library (no bloated dependencies)
🧠 Decision-making powered by the Minimax algorithm (zero-sum game theory)
🛡️ Robust input validation preventing invalid moves and board overwrites
📊 Clean 3×3 grid rendering with dynamic win/draw detection
🔄 Seamless replay loop for multiple matches

Building this project reinforced my understanding of recursive tree traversal, game state evaluation, and adversarial search in classical AI.

Check out the demo and code here! 👇

#Python #ArtificialIntelligence #CodeOrbitTech #Internship #Minimax #GameTheory #Coding #BTech #MachineLearning #ComputerScience
```

---

## 10. Short Interview Explanation (Q&A Preparation)

### 1. What did you build?
> *"I built a console-based Tic-Tac-Toe game in pure Python where a human player competes against an intelligent computer AI. The computer opponent uses the Minimax algorithm to evaluate every possible board state and play optimally."*

### 2. How does it work?
> *"The game represents the 3×3 grid as a 9-element list. Players take alternating turns choosing numbers 1 to 9. The program validates moves, updates the board display after each turn, checks 8 possible win combinations or a full board draw, and offers a replay loop once the match concludes."*

### 3. How does the AI make decisions?
> *"The AI uses the Minimax algorithm. It recursively simulates every possible future move down to terminal outcomes—win, loss, or draw. It assigns a numerical score to each outcome (+10 for AI win, -10 for Human win, 0 for draw), subtracts depth to favor faster wins, and selects the move that maximizes its score assuming the human also plays optimally."*

### 4. Why did you use Minimax?
> *"Because Tic-Tac-Toe is a two-player, turn-based, deterministic, zero-sum game with complete information. The state space is small (fewer than 9! states), making Minimax optimal and computationally instantaneous. It guarantees the AI will never lose."*

### 5. What did you learn?
> *"I deepened my understanding of adversarial search algorithms, recursive backtracking, evaluation functions with depth penalties, and writing clean, modular Python code with robust input validation and error handling."*
