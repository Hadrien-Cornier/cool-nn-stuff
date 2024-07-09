from chess_board import ChessBoardState
def create_test_position():
    board = ChessBoardState()
    board.board = [
        ['R', '.', '.', '.', 'K', '.', '.', 'R'],
        ['.', '.', 'P', '.', '.', 'P', 'P', '.'],
        ['.', '.', '.', '.', '.', 'N', '.', '.'],
        ['.', '.', '.', 'p', 'P', '.', '.', '.'],
        ['.', 'p', '.', '.', '.', '.', '.', '.'],
        ['.', '.', 'n', '.', '.', '.', '.', '.'],
        ['p', '.', 'p', 'q', '.', 'p', 'p', 'p'],
        ['r', '.', 'b', '.', 'k', 'b', 'n', 'r']
    ]
    board.to_move = 'w'
    board.castling_rights = 'K'
    board.en_passant_square = 'd6'
    return board

# Create and print the test position
test_board = create_test_position()
print("Test Position:")
for row in test_board.board:
    print(' '.join(row))
print(f"To move: {test_board.to_move}")
print(f"Castling rights: {test_board.castling_rights}")
print(f"En passant square: {test_board.en_passant_square}")

# Get legal moves
legal_moves = test_board.get_legal_moves()
print("\nLegal moves:")
print(legal_moves)
