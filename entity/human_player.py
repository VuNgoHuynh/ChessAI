"""HumanPlayer entity: turns mouse clicks into python-chess moves.

Click-to-move works in two clicks: the first click selects one of the player's
own pieces (the source square), the second click chooses a destination. When the
two squares form a legal move it is returned; otherwise the selection is updated
or cleared. Promotions are auto-queened.
"""

import chess


class HumanPlayer:
    def __init__(self, color=chess.BLACK):
        self.color = color
        self.click = None   # selected source square, or None
        self.move = None    # last completed move

    def handle_click(self, pos, board):
        """Process a mouse click at pixel `pos` against `board` (entity.Board).

        Returns a chess.Move when a full legal move is formed, else None.
        Also drives `board.selected` so the UI reflects the current selection.
        """
        square = board.square_at(pos)
        if square is None:
            return None

        piece = board.board.piece_at(square)

        # First click: pick up one of our own pieces.
        if self.click is None:
            if piece is not None and piece.color == self.color:
                self.click = square
                board.selected = square
            return None

        # Clicking the same square again cancels the selection.
        if square == self.click:
            self._clear(board)
            return None

        # Try to complete a move from the selected square to here.
        move = self._build_move(self.click, square, board.board)
        if move is not None:
            self.move = move
            self._clear(board)
            return move

        # Not a legal move: reselect if we clicked another of our pieces.
        if piece is not None and piece.color == self.color:
            self.click = square
            board.selected = square
        else:
            self._clear(board)
        return None

    def _build_move(self, frm, to, chess_board):
        move = chess.Move(frm, to)
        if move in chess_board.legal_moves:
            return move
        promo = chess.Move(frm, to, promotion=chess.QUEEN)
        if promo in chess_board.legal_moves:
            return promo
        return None

    def _clear(self, board):
        self.click = None
        board.selected = None
