"""HumanPlayer click-to-move flow (headless)."""

import chess

from entity.human_player import HumanPlayer


def _click_center(human, board, square):
    """Click the centre of `square` and return the resulting move (or None)."""
    x, y = board.pixel_of(square)
    return human.handle_click((x + 45, y + 45), board)


def test_two_click_move(make_board):
    board = make_board()  # standard start, White to move
    human = HumanPlayer(chess.WHITE)

    # First click selects the pawn.
    assert _click_center(human, board, chess.E2) is None
    assert board.selected == chess.E2

    # Second click on a legal destination completes the move.
    move = _click_center(human, board, chess.E4)
    assert move == chess.Move(chess.E2, chess.E4)
    assert board.selected is None


def test_clicking_own_piece_reselects(make_board):
    board = make_board()
    human = HumanPlayer(chess.WHITE)
    _click_center(human, board, chess.E2)
    # Clicking another of our own pawns moves the selection.
    assert _click_center(human, board, chess.D2) is None
    assert board.selected == chess.D2


def test_clicking_same_square_cancels(make_board):
    board = make_board()
    human = HumanPlayer(chess.WHITE)
    _click_center(human, board, chess.E2)
    assert _click_center(human, board, chess.E2) is None
    assert board.selected is None


def test_cannot_pick_up_opponent_piece(make_board):
    board = make_board()
    human = HumanPlayer(chess.WHITE)
    # e7 is a black pawn; White can't select it.
    assert _click_center(human, board, chess.E7) is None
    assert board.selected is None
