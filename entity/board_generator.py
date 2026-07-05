"""BoardGenerator entity: builds a ready-to-use Board.

Responsible for loading the piece images and sound effects (once, up front,
instead of per frame as Main.py did) and constructing a Board wired to a fresh
python-chess position. Asset paths are resolved relative to the repository root
so it works regardless of the current working directory.
"""

import os
import json
import random
import urllib.request

import pygame
import chess

from .board import Board, SQSIZE
from .sound import SoundEffect

# Repository root == parent of this entity package.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(ROOT, "Images")
SOUND_DIR = os.path.join(ROOT, "SoundEffect")

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


class BoardGenerator:
    """Factory that loads assets and produces a configured Board."""

    @staticmethod
    def load_images(size=SQSIZE):
        """Return {piece.symbol(): Surface} for all 12 pieces, scaled to `size`."""
        images = {}
        for color, prefix in ((chess.WHITE, "white"), (chess.BLACK, "black")):
            for piece_type, name in PIECE_NAMES.items():
                path = os.path.join(IMAGE_DIR, f"{prefix}-{name}.png")
                surface = pygame.transform.scale(
                    pygame.image.load(path), (size, size))
                symbol = chess.Piece(piece_type, color).symbol()
                images[symbol] = surface
        return images

    @staticmethod
    def load_sounds():
        """Return a SoundEffect loaded from the SoundEffect/ directory."""
        return SoundEffect.load(SOUND_DIR)

    # Where "Random" mode gets its puzzle from.
    PUZZLE_URL = "https://lichess.org/api/puzzle/next"

    @staticmethod
    def random_puzzle_fen(timeout=5):
        """Return the FEN of a real mid-game tactical puzzle, or ``None``.

        This is the "Random" game-mode seam: instead of a random *start*
        position it drops the player into a live puzzle fetched from the Lichess
        puzzle API. Lichess describes a puzzle as a game replayed up to
        ``initialPly`` plus a first "setup" move (``solution[0]``) that the
        opponent auto-plays; the position *after* that setup move — with the
        solver to move — is the tactic to solve, so that is what we load. The
        result is a **standard** chess position (load with ``chess960=False``).

        Returns ``None`` on any failure (network error, unexpected payload, an
        illegal setup move) so callers can fall back to another start position;
        the blocking network fetch uses a short ``timeout`` to keep the GUI
        responsive.
        """
        try:
            req = urllib.request.Request(
                BoardGenerator.PUZZLE_URL,
                headers={"User-Agent": "ChessAI/1.0 (pygame chess)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            pgn = data["game"]["pgn"]
            initial_ply = data["puzzle"]["initialPly"]
            solution = data["puzzle"]["solution"]

            # Replay the game up to and including ``initial_ply``; the opponent's
            # setup move is legal here (verified against the live API).
            board = chess.Board()
            for i, san in enumerate(pgn.split()):
                board.push_san(san)
                if i == initial_ply:
                    break
            setup = chess.Move.from_uci(solution[0])
            if setup not in board.legal_moves:
                return None
            board.push(setup)
            return board.fen()
        except Exception:
            return None

    @staticmethod
    def random_chess960_fen():
        """Return the FEN of a random Chess960 (Fischer-random) start position.

        Standard pawns, but the back-rank pieces are shuffled into one of the 960
        legal Fischer arrangements. The FEN must be loaded with ``chess960=True``
        (see ``generate``) so python-chess interprets the non-standard castling.
        Used as the offline fallback for "Random" mode when a puzzle can't be
        fetched (see ``random_puzzle_fen``).
        """
        return chess.Board.from_chess960_pos(random.randint(0, 959)).fen()

    @classmethod
    def generate(cls, screen, fen=None, flipped=False, chess960=False):
        """Build a Board on `screen`, optionally from a starting FEN.

        Pass ``flipped=True`` to mirror the board vertically so White sits at the
        bottom (used when the human plays White). Pass ``chess960=True`` when the
        FEN is a Fischer-random position so castling is interpreted correctly.
        """
        images = cls.load_images()
        sound = cls.load_sounds()
        chess_board = (chess.Board(fen, chess960=chess960) if fen
                       else chess.Board())
        return Board(screen, images, sound, chess_board, flipped=flipped)
