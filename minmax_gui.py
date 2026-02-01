"""
Tic-Tac-Toe GUI with Minimax and Alpha-Beta Pruning AI
"""
import tkinter as tk
from tkinter import ttk, messagebox
import math
import time

COLORS = {
    'bg': '#2C3E50',
    'board': '#34495E',
    'cell': '#ECF0F1',
    'cell_hover': '#BDC3C7',
    'x_color': '#E74C3C',
    'o_color': '#3498DB',
    'win_line': '#2ECC71',
    'text': '#FFFFFF'
}


class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe - Minimax & Alpha-Beta AI")
        self.root.geometry("500x650")
        self.root.configure(bg=COLORS['bg'])
        
        # Game state
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.current_player = 1  # 1 = X (Human), -1 = O (AI)
        self.game_over = False
        self.buttons = [[None]*3 for _ in range(3)]
        
        # Stats
        self.nodes_explored = 0
        self.pruned_nodes = 0
        self.ai_time = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="🎮 Tic-Tac-Toe AI 🎮",
                        font=('Arial', 22, 'bold'), bg=COLORS['bg'], fg=COLORS['text'])
        title.pack(pady=15)
        
        # Control frame
        control_frame = tk.Frame(self.root, bg=COLORS['bg'])
        control_frame.pack(pady=10)
        
        tk.Label(control_frame, text="AI Algorithm:", bg=COLORS['bg'], fg=COLORS['text'],
                font=('Arial', 11)).pack(side=tk.LEFT, padx=5)
        
        self.algo_var = tk.StringVar(value="Alpha-Beta")
        algo_combo = ttk.Combobox(control_frame, textvariable=self.algo_var,
                                  values=["Minimax", "Alpha-Beta"], state="readonly", width=12)
        algo_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Label(control_frame, text="Play as:", bg=COLORS['bg'], fg=COLORS['text'],
                font=('Arial', 11)).pack(side=tk.LEFT, padx=(20, 5))
        
        self.player_var = tk.StringVar(value="X (First)")
        player_combo = ttk.Combobox(control_frame, textvariable=self.player_var,
                                    values=["X (First)", "O (Second)"], state="readonly", width=10)
        player_combo.pack(side=tk.LEFT, padx=5)
        
        # New game button
        new_btn = tk.Button(control_frame, text="🔄 New Game", command=self.new_game,
                           bg='#27AE60', fg='white', font=('Arial', 10, 'bold'), padx=10)
        new_btn.pack(side=tk.LEFT, padx=15)
        
        # Game board frame
        board_frame = tk.Frame(self.root, bg=COLORS['board'], padx=10, pady=10)
        board_frame.pack(pady=20)
        
        for i in range(3):
            for j in range(3):
                btn = tk.Button(board_frame, text="", width=5, height=2,
                               font=('Arial', 36, 'bold'), bg=COLORS['cell'],
                               activebackground=COLORS['cell_hover'],
                               command=lambda r=i, c=j: self.make_move(r, c))
                btn.grid(row=i, column=j, padx=3, pady=3)
                btn.bind('<Enter>', lambda e, b=btn: b.config(bg=COLORS['cell_hover']) if b['text'] == '' else None)
                btn.bind('<Leave>', lambda e, b=btn: b.config(bg=COLORS['cell']) if b['text'] == '' else None)
                self.buttons[i][j] = btn
        
        # Status label
        self.status_label = tk.Label(self.root, text="Your turn (X)",
                                     font=('Arial', 14, 'bold'), bg=COLORS['bg'], fg=COLORS['text'])
        self.status_label.pack(pady=10)
        
        # Stats frame
        stats_frame = tk.LabelFrame(self.root, text="AI Statistics", bg=COLORS['bg'], 
                                    fg=COLORS['text'], font=('Arial', 11, 'bold'))
        stats_frame.pack(pady=10, padx=20, fill=tk.X)
        
        stats_inner = tk.Frame(stats_frame, bg=COLORS['bg'])
        stats_inner.pack(padx=10, pady=10)
        
        self.nodes_label = tk.Label(stats_inner, text="Nodes Explored: 0",
                                    font=('Arial', 10), bg=COLORS['bg'], fg=COLORS['text'])
        self.nodes_label.pack(side=tk.LEFT, padx=15)
        
        self.pruned_label = tk.Label(stats_inner, text="Nodes Pruned: 0",
                                     font=('Arial', 10), bg=COLORS['bg'], fg=COLORS['text'])
        self.pruned_label.pack(side=tk.LEFT, padx=15)
        
        self.time_label = tk.Label(stats_inner, text="AI Time: 0.000s",
                                   font=('Arial', 10), bg=COLORS['bg'], fg=COLORS['text'])
        self.time_label.pack(side=tk.LEFT, padx=15)
        
        # Score frame
        score_frame = tk.Frame(self.root, bg=COLORS['bg'])
        score_frame.pack(pady=10)
        
        self.x_score = 0
        self.o_score = 0
        self.draws = 0
        
        self.score_label = tk.Label(score_frame, text="X: 0  |  O: 0  |  Draws: 0",
                                    font=('Arial', 12, 'bold'), bg=COLORS['bg'], fg=COLORS['text'])
        self.score_label.pack()
        
        # Info
        info_frame = tk.LabelFrame(self.root, text="Algorithm Info", bg=COLORS['bg'],
                                   fg=COLORS['text'], font=('Arial', 10, 'bold'))
        info_frame.pack(pady=5, padx=20, fill=tk.X)
        
        info_text = "• Minimax: Explores entire game tree\n• Alpha-Beta: Prunes irrelevant branches for efficiency"
        tk.Label(info_frame, text=info_text, bg=COLORS['bg'], fg='#BDC3C7',
                font=('Arial', 9), justify=tk.LEFT).pack(padx=10, pady=5)
    
    def new_game(self):
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.game_over = False
        self.nodes_explored = 0
        self.pruned_nodes = 0
        
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text="", bg=COLORS['cell'], state=tk.NORMAL)
        
        # Set who plays first
        if "X" in self.player_var.get():
            self.current_player = 1
            self.status_label.config(text="Your turn (X)")
        else:
            self.current_player = 1  # X still goes first
            self.status_label.config(text="AI thinking...")
            self.root.after(500, self.ai_move)
        
        self.update_stats()
    
    def make_move(self, row, col):
        if self.game_over or self.board[row][col] != 0:
            return
        
        human_symbol = "X" if "X" in self.player_var.get() else "O"
        human_val = 1 if human_symbol == "X" else -1
        
        # Check if it's human's turn
        if self.current_player != human_val:
            return
        
        # Make human move
        self.board[row][col] = human_val
        self.buttons[row][col].config(text=human_symbol, 
                                       fg=COLORS['x_color'] if human_symbol == 'X' else COLORS['o_color'],
                                       bg=COLORS['cell'])
        
        # Check for game end
        if self.check_game_end():
            return
        
        # AI's turn
        self.current_player = -human_val
        self.status_label.config(text="AI thinking...")
        self.root.update()
        self.root.after(300, self.ai_move)
    
    def ai_move(self):
        ai_symbol = "O" if "X" in self.player_var.get() else "X"
        ai_val = -1 if ai_symbol == "O" else 1
        
        start_time = time.time()
        self.nodes_explored = 0
        self.pruned_nodes = 0
        
        # Get AI move
        if self.algo_var.get() == "Minimax":
            move = self.get_minimax_move(ai_val)
        else:
            move = self.get_alphabeta_move(ai_val)
        
        self.ai_time = time.time() - start_time
        self.update_stats()
        
        if move:
            row, col = move
            self.board[row][col] = ai_val
            self.buttons[row][col].config(text=ai_symbol,
                                          fg=COLORS['x_color'] if ai_symbol == 'X' else COLORS['o_color'],
                                          bg=COLORS['cell'])
        
        # Check for game end
        if self.check_game_end():
            return
        
        human_val = 1 if "X" in self.player_var.get() else -1
        self.current_player = human_val
        human_symbol = "X" if human_val == 1 else "O"
        self.status_label.config(text=f"Your turn ({human_symbol})")
    
    def get_available_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves
    
    def check_winner(self):
        # Check rows
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                return self.board[i][0], [(i, 0), (i, 1), (i, 2)]
        
        # Check columns
        for j in range(3):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] != 0:
                return self.board[0][j], [(0, j), (1, j), (2, j)]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[0][0], [(0, 0), (1, 1), (2, 2)]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[0][2], [(0, 2), (1, 1), (2, 0)]
        
        # Check draw
        if len(self.get_available_moves()) == 0:
            return 0, None
        
        return None, None
    
    def check_game_end(self):
        winner, winning_cells = self.check_winner()
        
        if winner is not None:
            self.game_over = True
            
            if winner == 0:
                self.status_label.config(text="It's a Draw!")
                self.draws += 1
            else:
                symbol = "X" if winner == 1 else "O"
                human_symbol = "X" if "X" in self.player_var.get() else "O"
                
                if symbol == human_symbol:
                    self.status_label.config(text=f"You Win! ({symbol})")
                    if symbol == "X":
                        self.x_score += 1
                    else:
                        self.o_score += 1
                else:
                    self.status_label.config(text=f"AI Wins! ({symbol})")
                    if symbol == "X":
                        self.x_score += 1
                    else:
                        self.o_score += 1
                
                # Highlight winning cells
                if winning_cells:
                    for r, c in winning_cells:
                        self.buttons[r][c].config(bg=COLORS['win_line'])
            
            self.score_label.config(text=f"X: {self.x_score}  |  O: {self.o_score}  |  Draws: {self.draws}")
            return True
        
        return False
    
    def evaluate(self, player):
        winner, _ = self.check_winner()
        if winner == player:
            return 10
        elif winner == -player:
            return -10
        return 0
    
    def get_minimax_move(self, player):
        best_score = -math.inf
        best_move = None
        
        for move in self.get_available_moves():
            self.board[move[0]][move[1]] = player
            score = self.minimax(0, False, player)
            self.board[move[0]][move[1]] = 0
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def minimax(self, depth, is_maximizing, player):
        self.nodes_explored += 1
        
        winner, _ = self.check_winner()
        if winner is not None:
            if winner == player:
                return 10 - depth
            elif winner == -player:
                return depth - 10
            return 0
        
        if is_maximizing:
            best_score = -math.inf
            for move in self.get_available_moves():
                self.board[move[0]][move[1]] = player
                score = self.minimax(depth + 1, False, player)
                self.board[move[0]][move[1]] = 0
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = math.inf
            for move in self.get_available_moves():
                self.board[move[0]][move[1]] = -player
                score = self.minimax(depth + 1, True, player)
                self.board[move[0]][move[1]] = 0
                best_score = min(score, best_score)
            return best_score
    
    def get_alphabeta_move(self, player):
        best_score = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf
        
        for move in self.get_available_moves():
            self.board[move[0]][move[1]] = player
            score = self.alphabeta(0, alpha, beta, False, player)
            self.board[move[0]][move[1]] = 0
            
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        
        return best_move
    
    def alphabeta(self, depth, alpha, beta, is_maximizing, player):
        self.nodes_explored += 1
        
        winner, _ = self.check_winner()
        if winner is not None:
            if winner == player:
                return 10 - depth
            elif winner == -player:
                return depth - 10
            return 0
        
        if is_maximizing:
            best_score = -math.inf
            for move in self.get_available_moves():
                self.board[move[0]][move[1]] = player
                score = self.alphabeta(depth + 1, alpha, beta, False, player)
                self.board[move[0]][move[1]] = 0
                best_score = max(score, best_score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    self.pruned_nodes += 1
                    break
            return best_score
        else:
            best_score = math.inf
            for move in self.get_available_moves():
                self.board[move[0]][move[1]] = -player
                score = self.alphabeta(depth + 1, alpha, beta, True, player)
                self.board[move[0]][move[1]] = 0
                best_score = min(score, best_score)
                beta = min(beta, score)
                if beta <= alpha:
                    self.pruned_nodes += 1
                    break
            return best_score
    
    def update_stats(self):
        self.nodes_label.config(text=f"Nodes Explored: {self.nodes_explored}")
        self.pruned_label.config(text=f"Nodes Pruned: {self.pruned_nodes}")
        self.time_label.config(text=f"AI Time: {self.ai_time:.4f}s")


if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()
