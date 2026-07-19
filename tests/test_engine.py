"""EnginePlayer (MCTS) move selection.

Serial tests seed ``random`` immediately before ``select_move`` so the
rollout-driven search is reproducible (the in-process path is otherwise
deterministic). The parallel path reseeds each worker from fresh OS entropy by
design, so its test asserts *structure* (a legal move) rather than a statistical
outcome. The multiprocessing test is skipped on Windows, where spawn-under-pytest
is unsafe; CI runs on Linux (fork), which exercises it.
"""

import random
import sys

import chess
import pytest

from entity.engine_player import EnginePlayer


def test_select_move_returns_legal(make_board):
    board = make_board()
    engine = EnginePlayer(chess.WHITE, num_simulation=120, num_workers=1)
    random.seed(12345)
    move = engine.select_move(board)
    assert move in board.board.legal_moves


def test_select_move_finds_mate_in_one(make_board):
    # Back-rank mate: the rook lift to e8 is checkmate.
    board = make_board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    engine = EnginePlayer(chess.WHITE, num_simulation=400, num_workers=1)
    random.seed(7)
    move = engine.select_move(board)
    after = board.board.copy()
    after.push(move)
    assert after.is_checkmate()


def test_select_move_none_when_game_over(make_board):
    board = make_board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")  # Black is checkmated
    assert board.board.is_game_over()
    engine = EnginePlayer(chess.BLACK, num_workers=1)
    assert engine.select_move(board) is None


def test_single_legal_move_is_forced(make_board):
    # Only the king move out of check is legal; the engine returns it directly.
    board = make_board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    engine = EnginePlayer(chess.WHITE, num_simulation=50, num_workers=1)
    legal = list(board.board.legal_moves)
    move = engine.select_move(board)
    assert move in legal


@pytest.mark.skipif(sys.platform == "win32",
                    reason="spawn-under-pytest is unsafe on Windows; CI (Linux) covers it")
def test_parallel_path_returns_legal(make_board):
    board = make_board()
    engine = EnginePlayer(chess.WHITE, num_simulation=60)  # default workers -> MP
    move = engine.select_move(board)
    assert move in board.board.legal_moves
