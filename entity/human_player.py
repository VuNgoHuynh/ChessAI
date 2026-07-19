"""HumanPlayer entity: turns mouse clicks into python-chess moves.

Click-to-move works in two clicks: the first click selects one of the player's
own pieces (the source square), the second click chooses a destination. When the
two squares form a legal move it is returned; otherwise the selection is updated
or cleared. When the destination is a promotion, a picker is opened on the board
instead — the next click chooses the piece (Queen/Rook/Bishop/Knight), or
cancels the move if it lands off the picker.
"""

import chess


class HumanPlayer:
    def __init__(self, color=chess.BLACK):
        self.color = color
        self.click = None       # selected source square, or None
        self.move = None        # last completed move
        self.promotion = None   # pending (from, to) awaiting a piece choice

    def handle_click(self, pos, board):
        """Process a mouse click at pixel `pos` against `board` (entity.Board).

        Returns a chess.Move when a full legal move is formed, else None.
        Also drives `board.selected` so the UI reflects the current selection.
        """
        # A promotion picker is open: this click chooses the piece (or cancels).
        if self.promotion is not None:
            return self._resolve_promotion(pos, board)

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

        # A promoting move: open the picker instead of completing it now.
        if self._is_promotion(self.click, square, board.board):
            self.promotion = (self.click, square)
            board.open_promotion(square, self.color)
            self.click = None
            board.selected = None
            return None

        # Try to complete a (non-promotion) move from the selection to here.
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

    def _resolve_promotion(self, pos, board):
        """Turn a click on the open picker into the promotion move (or cancel)."""
        frm, to = self.promotion
        choice = board.promotion_choice_at(pos)
        self.promotion = None
        board.promotion = None
        if choice is None:  # clicked off the picker: cancel the move
            return None
        move = chess.Move(frm, to, promotion=choice)
        if move in board.board.legal_moves:
            self.move = move
            return move
        return None

    @staticmethod
    def _is_promotion(frm, to, chess_board):
        """True if `frm`->`to` is a pawn promotion (last-rank arrival).

        Exact: a bare move is illegal there while the queen-promotion form is
        legal exactly when the pawn reaches the last rank.
        """
        return (chess.Move(frm, to) not in chess_board.legal_moves
                and chess.Move(frm, to, promotion=chess.QUEEN)
                in chess_board.legal_moves)

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
        self.promotion = None
        board.selected = None
