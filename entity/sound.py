"""SoundEffect entity: owns loading and playing the board's sound effects.

Previously this logic lived inline in Board (a `sounds` dict plus a `_play`
helper) and BoardGenerator (`load_sounds`). It is collected here so the board
only has to say *what* happened; this class decides *how* it sounds.
"""

import os

import pygame

# The five effects, matched to `SoundEffect/{name}.mp3`.
NAMES = ["Capture", "Castle", "Check", "Move", "Promote"]


class SoundEffect:
    """Loads the effect clips once and plays them by name."""

    def __init__(self, sounds=None):
        # {name: pygame.mixer.Sound}; empty when the mixer is unavailable.
        self.sounds = sounds or {}

    @classmethod
    def load(cls, sound_dir):
        """Build a SoundEffect from the .mp3 files in `sound_dir`.

        Initialises the mixer if needed; any clip that fails to load is simply
        skipped, so a missing file or headless mixer degrades to silence rather
        than raising.
        """
        sounds = {}
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
            except Exception:
                return cls(sounds)
        for name in NAMES:
            path = os.path.join(sound_dir, f"{name}.mp3")
            try:
                sounds[name] = pygame.mixer.Sound(path)
            except Exception:
                pass
        return cls(sounds)

    @staticmethod
    def classify(board, move):
        """Name the effect for `move` on `board`, judged *before* it is pushed.

        Returns one of "Castle", "Capture", "Promote" or "Move". Check is not
        decided here because it depends on the position *after* the move; the
        caller upgrades to "Check" once the move is on the stack.
        """
        if board.is_castling(move):
            return "Castle"
        if board.is_capture(move):
            return "Capture"
        if move.promotion:
            return "Promote"
        return "Move"

    def play(self, name):
        """Play the named effect; a no-op if it is missing or playback fails."""
        snd = self.sounds.get(name)
        if snd is not None:
            try:
                snd.play()
            except Exception:
                pass

    def __len__(self):
        return len(self.sounds)
