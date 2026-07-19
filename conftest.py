"""Pytest fixtures and headless setup shared by the whole suite.

Set the SDL dummy drivers *before* any test imports pygame, so the suite runs
without a display or audio device (CI and local headless runs alike).
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import chess       # noqa: E402  (import after env is configured)
import pygame      # noqa: E402
import pytest      # noqa: E402

from entity.board import Board  # noqa: E402


@pytest.fixture
def make_board():
    """Return a factory building an entity.Board on a headless dummy display.

    Self-healing about pygame state: it (re)initialises the display if a prior
    test tore it down (e.g. Main.main calls pygame.quit()), so tests are order
    independent. Images are empty — draw() tolerates missing art headless.
    """
    def _make(fen=None, flipped=False):
        if not pygame.get_init():
            pygame.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1, 1))
        chess_board = chess.Board(fen) if fen else chess.Board()
        return Board(pygame.display.get_surface(), {},
                     chess_board=chess_board, flipped=flipped)
    return _make
