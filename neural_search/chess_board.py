# Class that stores a chess board and its state

class ChessBoardState:
    def __init__(self):
        self.board = [['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
                      ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                      ['.', '.', '.', '.', '.', '.', '.', '.'],
                      ['.', '.', '.', '.', '.', '.', '.', '.'],
                      ['.', '.', '.', '.', '.', '.', '.', '.'],
                      ['.', '.', '.', '.', '.', '.', '.', '.'],
                      ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
                      ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']]
        self.to_move = 'w'
        self.castling_rights = 'KQkq'
        self.en_passant_square = '-'

    def get_legal_moves(self):
        legal_moves = []
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece.isupper() and self.to_move == 'w':
                    legal_moves.extend(self.get_moves_for_piece(i, j, piece))
                elif piece.islower() and self.to_move == 'b':
                    legal_moves.extend(self.get_moves_for_piece(i, j, piece))
        return legal_moves

    def get_moves_for_piece(self, i, j, piece):
        moves = []
        if piece == 'R' or piece == 'r':
            moves.extend(self.get_rook_moves(i, j, piece))
        elif piece == 'N' or piece == 'n':
            moves.extend(self.get_knight_moves(i, j, piece))
        elif piece == 'B' or piece == 'b':
            moves.extend(self.get_bishop_moves(i, j, piece))
        elif piece == 'Q' or piece == 'q':
            moves.extend(self.get_queen_moves(i, j, piece))
        elif piece == 'K' or piece == 'k':
            moves.extend(self.get_king_moves(i, j, piece))
        elif piece == 'P' or piece == 'p':
            moves.extend(self.get_pawn_moves(i, j, piece))
        return moves

    def get_rook_moves(self, i, j, piece):
        moves = []
        for x in range(8):
            if x != j:
                moves.append((piece,i, x))
        for y in range(8):
            if y != i:
                moves.append((piece,y, j))
        return moves

    def get_knight_moves(self, i, j, piece):
        moves = []
        for x in range(-2, 3):
            for y in range(-2, 3):
                if abs(x) + abs(y) == 3:
                    new_i = i + x
                    new_j = j + y
                    if 0 <= new_i < 8 and 0 <= new_j < 8:
                        moves.append((piece,new_i, new_j))
        return moves

    def get_bishop_moves(self, i, j, piece):
        moves = []
        for x in range(-7, 8):
            for y in range(-7, 8):
                if abs(x) == abs(y):
                    new_i = i + x
                    new_j = j + y
                    if 0 <= new_i < 8 and 0 <= new_j < 8:
                        moves.append((piece,new_i, new_j))
        return moves

    def get_queen_moves(self, i, j, piece):
        moves = []
        for x in range(-7, 8):
            for y in range(-7, 8):
                if abs(x) == abs(y) or x == 0 or y == 0:
                    new_i = i + x
                    new_j = j + y
                    if 0 <= new_i < 8 and 0 <= new_j < 8:
                        moves.append((piece,new_i, new_j))
        return moves

    def get_king_moves(self, i, j, piece):
        moves = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                new_i = i + x
                new_j = j + y
                if 0 <= new_i < 8 and 0 <= new_j < 8:
                    moves.append((piece,new_i, new_j))
        return moves

    def get_pawn_moves(self, i, j, piece):
        moves = []
        if piece == 'P':
            if i == 1:
                moves.append((piece,i + 2, j))
            moves.append((piece,i + 1, j))
        elif piece == 'p':
            if i == 6:
                moves.append((piece,i - 2, j))
            moves.append((piece,i - 1, j))
        return moves

    def make_move(self, move):
        pass

    def is_checkmate(self):
        pass
