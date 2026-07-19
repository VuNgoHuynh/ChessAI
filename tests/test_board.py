"""Board coordinate system and rendering (headless)."""

import chess


def test_roundtrip_not_flipped(make_board):
    board = make_board(flipped=False)
    for sq in chess.SQUARES:
        assert board.square_at(board.pixel_of(sq)) == sq
    # Not flipped: White at the top, A1 at the top-left (row 0).
    assert board.square_at((0, 0)) == chess.A1
    assert board.pixel_of(chess.A1) == (0, 0)
    assert board.square_at((0, 630)) == chess.A8


def test_roundtrip_flipped(make_board):
    board = make_board(flipped=True)
    for sq in chess.SQUARES:
        assert board.square_at(board.pixel_of(sq)) == sq
    # Flipped: White at the bottom, so A1 sits at the bottom-left.
    assert board.square_at((0, 0)) == chess.A8
    assert board.pixel_of(chess.A1) == (0, 630)
    assert board.square_at((0, 630)) == chess.A1


def test_square_at_off_board(make_board):
    board = make_board()
    assert board.square_at((-1, 0)) is None
    assert board.square_at((720, 0)) is None  # past the 8 board columns


def test_target_squares_from_start(make_board):
    board = make_board()
    # White knight on b1 can reach a3 and c3 at the start.
    assert set(board.target_squares(chess.B1)) == {chess.A3, chess.C3}
    # A blocked piece (rook a1) has no legal targets.
    assert board.target_squares(chess.A1) == []


def test_draw_headless(make_board):
    # Rendering (squares, pieces, panel, game-over veil) must not raise headless.
    make_board().draw()
    make_board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").draw()  # a game-over position
