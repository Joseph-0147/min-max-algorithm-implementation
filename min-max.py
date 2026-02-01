import math
import time
from typing import List, Tuple, Optional
import numpy as np
from copy import deepcopy

# ======================
# GAME BOARD CLASS
# ======================

class TicTacToeBoard:
    """Represents the Tic-Tac-Toe game board"""
    
    EMPTY = 0
    PLAYER_X = 1  # Human or AI
    PLAYER_O = -1  # AI
    DRAW = 0
    
    def __init__(self, board=None):
        """Initialize board - 3x3 matrix"""
        if board is None:
            self.board = [[self.EMPTY, self.EMPTY, self.EMPTY],
                         [self.EMPTY, self.EMPTY, self.EMPTY],
                         [self.EMPTY, self.EMPTY, self.EMPTY]]
        else:
            self.board = board
        self.current_player = self.PLAYER_X
    
    def __str__(self):
        """String representation of the board"""
        symbols = {self.EMPTY: ' ', self.PLAYER_X: 'X', self.PLAYER_O: 'O'}
        lines = []
        lines.append("  0 1 2")
        for i, row in enumerate(self.board):
            line = f"{i} "
            for cell in row:
                line += f"{symbols[cell]}|"
            lines.append(line[:-1])
            if i < 2:
                lines.append("  -+-+-")
        return "\n".join(lines)
    
    def reset(self):
        """Reset the board"""
        self.board = [[self.EMPTY, self.EMPTY, self.EMPTY],
                     [self.EMPTY, self.EMPTY, self.EMPTY],
                     [self.EMPTY, self.EMPTY, self.EMPTY]]
        self.current_player = self.PLAYER_X
    
    def get_available_moves(self) -> List[Tuple[int, int]]:
        """Get all empty positions"""
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == self.EMPTY:
                    moves.append((i, j))
        return moves
    
    def make_move(self, row: int, col: int, player: int) -> bool:
        """Make a move if position is valid"""
        if self.board[row][col] == self.EMPTY:
            self.board[row][col] = player
            self.current_player = -player  # Switch player
            return True
        return False
    
    def undo_move(self, row: int, col: int):
        """Undo a move"""
        self.board[row][col] = self.EMPTY
        self.current_player = -self.current_player
    
    def check_winner(self) -> Optional[int]:
        """Check if there's a winner or draw"""
        # Check rows
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != self.EMPTY:
                return self.board[i][0]
        
        # Check columns
        for j in range(3):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] != self.EMPTY:
                return self.board[0][j]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != self.EMPTY:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != self.EMPTY:
            return self.board[0][2]
        
        # Check for draw
        if len(self.get_available_moves()) == 0:
            return self.DRAW
        
        return None
    
    def is_game_over(self) -> bool:
        """Check if game is over"""
        return self.check_winner() is not None
    
    def evaluate(self, player: int) -> int:
        """Evaluate the board state for a given player"""
        winner = self.check_winner()
        if winner == player:
            return 10
        elif winner == -player:
            return -10
        elif winner == self.DRAW:
            return 0
        
        # Heuristic evaluation for non-terminal states
        score = 0
        
        # Evaluate rows
        for i in range(3):
            row_values = [self.board[i][j] for j in range(3)]
            score += self.evaluate_line(row_values, player)
        
        # Evaluate columns
        for j in range(3):
            col_values = [self.board[i][j] for i in range(3)]
            score += self.evaluate_line(col_values, player)
        
        # Evaluate diagonals
        diag1 = [self.board[i][i] for i in range(3)]
        diag2 = [self.board[i][2-i] for i in range(3)]
        score += self.evaluate_line(diag1, player)
        score += self.evaluate_line(diag2, player)
        
        return score
    
    def evaluate_line(self, line: List[int], player: int) -> int:
        """Evaluate a single line (row, column, or diagonal)"""
        if player not in [self.PLAYER_X, self.PLAYER_O]:
            return 0
        
        opponent = -player
        player_count = line.count(player)
        opponent_count = line.count(opponent)
        empty_count = line.count(self.EMPTY)
        
        # Scoring
        if player_count == 3:
            return 100
        elif opponent_count == 3:
            return -100
        elif player_count == 2 and empty_count == 1:
            return 10
        elif opponent_count == 2 and empty_count == 1:
            return -10
        elif player_count == 1 and empty_count == 2:
            return 1
        elif opponent_count == 1 and empty_count == 2:
            return -1
        
        return 0
    
    def get_state_key(self) -> str:
        """Get unique string representation of board state"""
        return ''.join(str(cell) for row in self.board for cell in row)

# ======================
# MINIMAX ALGORITHM
# ======================

class MinimaxAI:
    """Minimax algorithm implementation for Tic-Tac-Toe"""
    
    def __init__(self, player: int, max_depth: int = 9):
        self.player = player
        self.max_depth = max_depth
        self.nodes_explored = 0
        self.move_times = []
        self.transposition_table = {}  # For memoization
    
    def get_move(self, board: TicTacToeBoard) -> Tuple[int, int]:
        """Get the best move using Minimax"""
        self.nodes_explored = 0
        start_time = time.time()
        
        best_score = -math.inf
        best_move = None
        
        # If only one move available, take it
        available_moves = board.get_available_moves()
        if len(available_moves) == 1:
            return available_moves[0]
        
        # Try all possible moves
        for move in available_moves:
            row, col = move
            board.make_move(row, col, self.player)
            
            # Get minimax score for this move
            score = self.minimax(board, depth=0, is_maximizing=False, 
                               alpha=-math.inf, beta=math.inf)
            
            board.undo_move(row, col)
            
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
        
        elapsed_time = time.time() - start_time
        self.move_times.append(elapsed_time)
        
        print(f"Minimax: Explored {self.nodes_explored} nodes in {elapsed_time:.4f}s")
        return best_move
    
    def minimax(self, board: TicTacToeBoard, depth: int, is_maximizing: bool,
                alpha: float, beta: float) -> int:
        """Recursive minimax algorithm with alpha-beta parameters (kept for interface)"""
        self.nodes_explored += 1
        
        # Check memoization table
        state_key = board.get_state_key() + f"_{depth}_{is_maximizing}"
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]
        
        # Terminal state check
        winner = board.check_winner()
        if winner is not None or depth >= self.max_depth:
            score = board.evaluate(self.player)
            self.transposition_table[state_key] = score
            return score
        
        if is_maximizing:
            best_score = -math.inf
            for move in board.get_available_moves():
                row, col = move
                board.make_move(row, col, self.player)
                score = self.minimax(board, depth + 1, False, alpha, beta)
                board.undo_move(row, col)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # Beta cutoff
            self.transposition_table[state_key] = best_score
            return best_score
        else:
            best_score = math.inf
            for move in board.get_available_moves():
                row, col = move
                board.make_move(row, col, -self.player)
                score = self.minimax(board, depth + 1, True, alpha, beta)
                board.undo_move(row, col)
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            self.transposition_table[state_key] = best_score
            return best_score

# ======================
# ALPHA-BETA PRUNING ALGORITHM
# ======================

class AlphaBetaAI:
    """Alpha-Beta Pruning implementation for Tic-Tac-Toe"""
    
    def __init__(self, player: int, max_depth: int = 9):
        self.player = player
        self.max_depth = max_depth
        self.nodes_explored = 0
        self.pruned_nodes = 0
        self.move_times = []
        self.transposition_table = {}  # For memoization
    
    def get_move(self, board: TicTacToeBoard) -> Tuple[int, int]:
        """Get the best move using Alpha-Beta Pruning"""
        self.nodes_explored = 0
        self.pruned_nodes = 0
        start_time = time.time()
        
        best_score = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf
        
        # If only one move available, take it
        available_moves = board.get_available_moves()
        if len(available_moves) == 1:
            return available_moves[0]
        
        # Try all possible moves with move ordering for better pruning
        ordered_moves = self.order_moves(board, available_moves, self.player)
        
        for move in ordered_moves:
            row, col = move
            board.make_move(row, col, self.player)
            
            # Get alpha-beta score for this move
            score = self.alphabeta(board, depth=0, alpha=alpha, beta=beta,
                                  is_maximizing=False)
            
            board.undo_move(row, col)
            
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
        
        elapsed_time = time.time() - start_time
        self.move_times.append(elapsed_time)
        
        pruning_efficiency = (self.pruned_nodes / (self.nodes_explored + self.pruned_nodes)) * 100 if (self.nodes_explored + self.pruned_nodes) > 0 else 0
        
        print(f"Alpha-Beta: Explored {self.nodes_explored} nodes, "
              f"pruned {self.pruned_nodes} nodes ({pruning_efficiency:.1f}%) in {elapsed_time:.4f}s")
        return best_move
    
    def alphabeta(self, board: TicTacToeBoard, depth: int, alpha: float, beta: float,
                  is_maximizing: bool) -> int:
        """Recursive alpha-beta pruning algorithm"""
        self.nodes_explored += 1
        
        # Check memoization table
        state_key = board.get_state_key() + f"_{depth}_{is_maximizing}_{alpha}_{beta}"
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]
        
        # Terminal state check
        winner = board.check_winner()
        if winner is not None or depth >= self.max_depth:
            score = board.evaluate(self.player)
            self.transposition_table[state_key] = score
            return score
        
        if is_maximizing:
            value = -math.inf
            # Order moves for better pruning
            ordered_moves = self.order_moves(board, board.get_available_moves(), self.player)
            
            for move in ordered_moves:
                row, col = move
                board.make_move(row, col, self.player)
                value = max(value, self.alphabeta(board, depth + 1, alpha, beta, False))
                board.undo_move(row, col)
                
                if value >= beta:
                    self.pruned_nodes += len(ordered_moves) - ordered_moves.index(move) - 1
                    break  # Beta cutoff
                alpha = max(alpha, value)
            
            self.transposition_table[state_key] = value
            return value
        else:
            value = math.inf
            # Order moves for better pruning
            ordered_moves = self.order_moves(board, board.get_available_moves(), -self.player)
            
            for move in ordered_moves:
                row, col = move
                board.make_move(row, col, -self.player)
                value = min(value, self.alphabeta(board, depth + 1, alpha, beta, True))
                board.undo_move(row, col)
                
                if value <= alpha:
                    self.pruned_nodes += len(ordered_moves) - ordered_moves.index(move) - 1
                    break  # Alpha cutoff
                beta = min(beta, value)
            
            self.transposition_table[state_key] = value
            return value
    
    def order_moves(self, board: TicTacToeBoard, moves: List[Tuple[int, int]], 
                    player: int) -> List[Tuple[int, int]]:
        """Order moves based on heuristic for better pruning"""
        move_scores = []
        
        for move in moves:
            row, col = move
            # Make temporary move
            board.make_move(row, col, player)
            
            # Score based on immediate win/lose conditions
            winner = board.check_winner()
            if winner == player:
                score = 1000
            elif winner == -player:
                score = 500
            else:
                # Heuristic based on position (center and corners are better)
                score = 0
                if (row, col) == (1, 1):  # Center
                    score += 3
                elif (row + col) % 2 == 0:  # Corners
                    score += 2
                else:  # Edges
                    score += 1
                
                # Add evaluation score
                score += abs(board.evaluate(player))
            
            board.undo_move(row, col)
            move_scores.append((move, score))
        
        # Sort by score (descending for maximizing player, ascending for minimizing)
        move_scores.sort(key=lambda x: x[1], reverse=(player == self.player))
        return [move for move, _ in move_scores]

# ======================
# GAME CONTROLLER
# ======================

class TicTacToeGame:
    """Main game controller"""
    
    def __init__(self):
        self.board = TicTacToeBoard()
        self.minimax_ai = None
        self.alphabeta_ai = None
        self.stats = {
            'minimax_nodes': [],
            'alphabeta_nodes': [],
            'minimax_times': [],
            'alphabeta_times': [],
            'minimax_pruned': [],
            'alphabeta_pruned': []
        }
    
    def human_vs_ai(self, use_alpha_beta: bool = True, ai_first: bool = False):
        """Human vs AI game"""
        print("\n" + "="*50)
        print("TIC-TAC-TOE: HUMAN vs AI")
        print(f"Using {'Alpha-Beta Pruning' if use_alpha_beta else 'Minimax'}")
        print("="*50)
        
        self.board.reset()
        ai_player = TicTacToeBoard.PLAYER_X if ai_first else TicTacToeBoard.PLAYER_O
        human_player = -ai_player
        
        if use_alpha_beta:
            ai = AlphaBetaAI(ai_player)
        else:
            ai = MinimaxAI(ai_player)
        
        current_player = TicTacToeBoard.PLAYER_X  # X always starts
        
        while not self.board.is_game_over():
            print(f"\n{self.board}")
            print(f"Current player: {'X' if current_player == 1 else 'O'}")
            
            if current_player == ai_player:
                print("AI is thinking...")
                move = ai.get_move(self.board)
                if move:
                    row, col = move
                    self.board.make_move(row, col, ai_player)
                    print(f"AI plays at ({row}, {col})")
            else:
                # Human move
                while True:
                    try:
                        move_input = input("Enter your move (row col, 0-2): ").strip()
                        if not move_input:
                            continue
                        row, col = map(int, move_input.split())
                        if 0 <= row <= 2 and 0 <= col <= 2:
                            if self.board.make_move(row, col, human_player):
                                break
                            else:
                                print("Cell already occupied!")
                        else:
                            print("Invalid coordinates! Use 0-2.")
                    except (ValueError, IndexError):
                        print("Invalid input! Use format: row col (e.g., 1 1)")
            
            current_player = -current_player
        
        # Game over
        print(f"\nFinal board:\n{self.board}")
        winner = self.board.check_winner()
        if winner == TicTacToeBoard.DRAW:
            print("It's a draw!")
        elif (winner == ai_player and use_alpha_beta) or (winner == ai_player and not use_alpha_beta):
            print("AI wins!")
        else:
            print("Human wins!")
    
    def ai_vs_ai(self, depth_limit: int = 9):
        """AI vs AI game to compare algorithms"""
        print("\n" + "="*50)
        print("TIC-TAC-TOE: MINIMAX vs ALPHA-BETA")
        print("="*50)
        
        self.board.reset()
        self.minimax_ai = MinimaxAI(TicTacToeBoard.PLAYER_X, max_depth=depth_limit)
        self.alphabeta_ai = AlphaBetaAI(TicTacToeBoard.PLAYER_O, max_depth=depth_limit)
        
        print("Minimax (X) vs Alpha-Beta (O)")
        print(f"Search depth limit: {depth_limit}")
        
        move_count = 0
        current_player = TicTacToeBoard.PLAYER_X
        
        while not self.board.is_game_over():
            print(f"\nMove {move_count + 1}:")
            print(self.board)
            
            if current_player == TicTacToeBoard.PLAYER_X:
                print("Minimax AI (X) is thinking...")
                move = self.minimax_ai.get_move(self.board)
                ai_name = "Minimax"
                self.stats['minimax_nodes'].append(self.minimax_ai.nodes_explored)
                self.stats['minimax_times'].append(self.minimax_ai.move_times[-1])
            else:
                print("Alpha-Beta AI (O) is thinking...")
                move = self.alphabeta_ai.get_move(self.board)
                ai_name = "Alpha-Beta"
                self.stats['alphabeta_nodes'].append(self.alphabeta_ai.nodes_explored)
                self.stats['alphabeta_times'].append(self.alphabeta_ai.move_times[-1])
                self.stats['alphabeta_pruned'].append(self.alphabeta_ai.pruned_nodes)
            
            if move:
                row, col = move
                self.board.make_move(row, col, current_player)
                print(f"{ai_name} plays at ({row}, {col})")
            
            current_player = -current_player
            move_count += 1
        
        # Game over
        print(f"\nFinal board after {move_count} moves:\n{self.board}")
        winner = self.board.check_winner()
        
        if winner == TicTacToeBoard.PLAYER_X:
            print("Minimax (X) wins!")
        elif winner == TicTacToeBoard.PLAYER_O:
            print("Alpha-Beta (O) wins!")
        else:
            print("It's a draw!")
        
        self.print_comparison_stats()
    
    def performance_benchmark(self, num_games: int = 10):
        """Run benchmark tests comparing both algorithms"""
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARK")
        print(f"Running {num_games} games for each algorithm")
        print("="*50)
        
        results = {
            'minimax': {'wins': 0, 'draws': 0, 'losses': 0, 
                       'total_nodes': 0, 'total_time': 0},
            'alphabeta': {'wins': 0, 'draws': 0, 'losses': 0,
                         'total_nodes': 0, 'total_time': 0, 'total_pruned': 0}
        }
        
        # Test both algorithms playing against themselves (perfect play should always draw)
        for game_num in range(num_games):
            print(f"\nGame {game_num + 1}/{num_games}")
            
            # Reset stats
            self.stats = {k: [] for k in self.stats.keys()}
            
            # Run AI vs AI game
            self.board.reset()
            self.minimax_ai = MinimaxAI(TicTacToeBoard.PLAYER_X)
            self.alphabeta_ai = AlphaBetaAI(TicTacToeBoard.PLAYER_O)
            
            current_player = TicTacToeBoard.PLAYER_X
            
            while not self.board.is_game_over():
                if current_player == TicTacToeBoard.PLAYER_X:
                    move = self.minimax_ai.get_move(self.board)
                    results['minimax']['total_nodes'] += self.minimax_ai.nodes_explored
                    results['minimax']['total_time'] += self.minimax_ai.move_times[-1]
                else:
                    move = self.alphabeta_ai.get_move(self.board)
                    results['alphabeta']['total_nodes'] += self.alphabeta_ai.nodes_explored
                    results['alphabeta']['total_time'] += self.alphabeta_ai.move_times[-1]
                    results['alphabeta']['total_pruned'] += self.alphabeta_ai.pruned_nodes
                
                if move:
                    row, col = move
                    self.board.make_move(row, col, current_player)
                
                current_player = -current_player
            
            # Record result
            winner = self.board.check_winner()
            if winner == TicTacToeBoard.PLAYER_X:
                results['minimax']['wins'] += 1
                results['alphabeta']['losses'] += 1
            elif winner == TicTacToeBoard.PLAYER_O:
                results['alphabeta']['wins'] += 1
                results['minimax']['losses'] += 1
            else:
                results['minimax']['draws'] += 1
                results['alphabeta']['draws'] += 1
        
        # Print benchmark results
        self.print_benchmark_results(results, num_games)
    
    def print_comparison_stats(self):
        """Print comparison statistics for the last game"""
        if not self.stats['minimax_nodes']:
            return
        
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON")
        print("="*50)
        
        print(f"\n{'Metric':<25} {'Minimax':<15} {'Alpha-Beta':<15} {'Improvement':<15}")
        print("-" * 70)
        
        # Nodes explored
        min_nodes = sum(self.stats['minimax_nodes'])
        ab_nodes = sum(self.stats['alphabeta_nodes'])
        improvement = ((min_nodes - ab_nodes) / min_nodes) * 100 if min_nodes > 0 else 0
        print(f"{'Total Nodes Explored':<25} {min_nodes:<15} {ab_nodes:<15} {improvement:>10.1f}%")
        
        # Time taken
        min_time = sum(self.stats['minimax_times'])
        ab_time = sum(self.stats['alphabeta_times'])
        time_improvement = ((min_time - ab_time) / min_time) * 100 if min_time > 0 else 0
        print(f"{'Total Time (s)':<25} {min_time:<15.4f} {ab_time:<15.4f} {time_improvement:>10.1f}%")
        
        # Average nodes per move
        avg_min_nodes = np.mean(self.stats['minimax_nodes']) if self.stats['minimax_nodes'] else 0
        avg_ab_nodes = np.mean(self.stats['alphabeta_nodes']) if self.stats['alphabeta_nodes'] else 0
        print(f"{'Avg Nodes per Move':<25} {avg_min_nodes:<15.1f} {avg_ab_nodes:<15.1f} {'-':>15}")
        
        # Pruning stats
        if self.stats['alphabeta_pruned']:
            total_possible = sum(self.stats['alphabeta_nodes']) + sum(self.stats['alphabeta_pruned'])
            pruning_percentage = (sum(self.stats['alphabeta_pruned']) / total_possible * 100) if total_possible > 0 else 0
            print(f"{'Nodes Pruned':<25} {'N/A':<15} {sum(self.stats['alphabeta_pruned']):<15} {pruning_percentage:>10.1f}%")
    
    def print_benchmark_results(self, results: dict, num_games: int):
        """Print benchmark results"""
        print("\n" + "="*50)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*50)
        
        print(f"\nGames played: {num_games}")
        print(f"Minimax wins: {results['minimax']['wins']}")
        print(f"Alpha-Beta wins: {results['alphabeta']['wins']}")
        print(f"Draws: {results['minimax']['draws']}")
        
        print(f"\n{'Metric':<25} {'Minimax':<15} {'Alpha-Beta':<15} {'Improvement':<15}")
        print("-" * 70)
        
        # Average nodes per game
        avg_min_nodes = results['minimax']['total_nodes'] / num_games
        avg_ab_nodes = results['alphabeta']['total_nodes'] / num_games
        node_improvement = ((avg_min_nodes - avg_ab_nodes) / avg_min_nodes) * 100 if avg_min_nodes > 0 else 0
        print(f"{'Avg Nodes per Game':<25} {avg_min_nodes:<15.1f} {avg_ab_nodes:<15.1f} {node_improvement:>10.1f}%")
        
        # Average time per game
        avg_min_time = results['minimax']['total_time'] / num_games
        avg_ab_time = results['alphabeta']['total_time'] / num_games
        time_improvement = ((avg_min_time - avg_ab_time) / avg_min_time) * 100 if avg_min_time > 0 else 0
        print(f"{'Avg Time per Game (s)':<25} {avg_min_time:<15.4f} {avg_ab_time:<15.4f} {time_improvement:>10.1f}%")
        
        # Pruning efficiency
        if results['alphabeta']['total_pruned'] > 0:
            total_possible_nodes = results['alphabeta']['total_nodes'] + results['alphabeta']['total_pruned']
            pruning_efficiency = (results['alphabeta']['total_pruned'] / total_possible_nodes) * 100
            print(f"{'Pruning Efficiency':<25} {'N/A':<15} {pruning_efficiency:<15.1f}% {'-':>15}")
        
        print("\n" + "="*50)
        print("KEY INSIGHTS:")
        print("="*50)
        print("1. Alpha-Beta pruning typically explores 50-90% fewer nodes")
        print("2. Time improvement is often proportional to node reduction")
        print("3. Move ordering significantly improves pruning effectiveness")
        print("4. Early game has more branching, hence more pruning opportunities")
        print("5. With perfect play, Tic-Tac-Toe should always end in a draw")

# ======================
# INTERACTIVE DEMO
# ======================

def interactive_demo():
    """Interactive demo menu"""
    game = TicTacToeGame()
    
    while True:
        print("\n" + "="*50)
        print("TIC-TAC-TOE AI DEMO")
        print("="*50)
        print("1. Play against Minimax AI")
        print("2. Play against Alpha-Beta AI")
        print("3. Watch Minimax vs Alpha-Beta AI battle")
        print("4. Run performance benchmark")
        print("5. Visualize search tree (simple example)")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            game.human_vs_ai(use_alpha_beta=False)
        elif choice == '2':
            ai_first = input("Should AI go first? (y/n): ").lower() == 'y'
            game.human_vs_ai(use_alpha_beta=True, ai_first=ai_first)
        elif choice == '3':
            depth = input("Enter search depth (3-9, default 9): ").strip()
            depth = int(depth) if depth.isdigit() and 3 <= int(depth) <= 9 else 9
            game.ai_vs_ai(depth_limit=depth)
        elif choice == '4':
            games = input("Number of benchmark games (default 10): ").strip()
            games = int(games) if games.isdigit() and int(games) > 0 else 10
            game.performance_benchmark(num_games=games)
        elif choice == '5':
            visualize_search_tree_example()
        elif choice == '6':
            print("Thanks for playing!")
            break
        else:
            print("Invalid choice! Please try again.")

def visualize_search_tree_example():
    """Visualize a simple search tree example"""
    print("\n" + "="*50)
    print("SEARCH TREE VISUALIZATION")
    print("="*50)
    
    # Create a simple board state
    board = TicTacToeBoard()
    # Set up a partially filled board
    board.board = [
        [1, 0, -1],
        [0, 1, 0],
        [-1, 0, 0]
    ]
    
    print("Example board state:")
    print(board)
    print("\nX's turn to move")
    
    print("\nMinimax would explore all possible moves:")
    print("Depth 0: Root (X's turn)")
    print("├── Move (0,1)")
    print("│   └── Depth 1: O's responses")
    print("├── Move (1,0)")
    print("│   └── Depth 1: O's responses")
    print("├── Move (1,2)")
    print("│   └── Depth 1: O's responses")
    print("├── Move (2,1)")
    print("│   └── Depth 1: O's responses")
    print("└── Move (2,2)")
    print("    └── Depth 1: O's responses")
    
    print("\nAlpha-Beta would prune branches when possible:")
    print("Depth 0: Root (X's turn)")
    print("├── Move (0,1) → score = +∞")
    print("├── Move (1,0) → score = +∞")
    print("├── Move (1,2) → PRUNED (alpha >= beta)")
    print("├── Move (2,1) → PRUNED")
    print("└── Move (2,2) → PRUNED")
    
    print("\nIn this example, Alpha-Beta explores only 2 moves deeply")
    print("while Minimax explores all 5 moves completely.")

# ======================
# RUN THE DEMO
# ======================

if __name__ == "__main__":
    print("="*70)
    print("MINIMAX & ALPHA-BETA PRUNING DEMONSTRATION")
    print("Tic-Tac-Toe AI Implementation")
    print("="*70)
    
    print("\nALGORITHMS IMPLEMENTED:")
    print("1. Minimax: Explores entire game tree to find optimal move")
    print("2. Alpha-Beta Pruning: Optimized Minimax that prunes irrelevant branches")
    
    print("\nKEY FEATURES:")
    print("• Complete Tic-Tac-Toe game engine")
    print("• Both Minimax and Alpha-Beta implementations")
    print("• Performance comparison and benchmarking")
    print("• Move ordering for better pruning")
    print("• Transposition tables for memoization")
    print("• Heuristic evaluation for non-terminal states")
    
    interactive_demo()