"""End-to-end smoke of the game loop, headless.

``player_color=WHITE`` means the engine is Black and it is the human's turn, so
no MCTS move is triggered within the frame budget — the loop just draws and
processes events. That keeps the smoke fast while still exercising the real
``main`` path (menus skipped, board built, game-over gate).
"""

import chess

import Main


def test_main_loop_runs_headless():
    Main.ENGINE_DELAY = 0
    board = Main.main(max_frames=4, player_color=chess.WHITE, mode="classic")
    assert board is not None
    assert not board.board.is_game_over()
