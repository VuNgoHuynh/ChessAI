"""Board entity: wraps a python-chess board with the pygame UI from Main.py.

Orientation is chosen so the **human always sits at the bottom** of the screen:

* Not flipped (``flipped=False``, the default — human plays Black): White is at
  the TOP (rank 1 at row 0), Black at the BOTTOM (rank 8 at row 7). This matches
  the original layout in Main.py.
* Flipped (``flipped=True`` — human plays White): the board is mirrored
  vertically so White (rank 1) sits at the BOTTOM and Black (rank 8) at the top.

Only the rank axis is mirrored; files stay in a-to-h left-to-right order in both
orientations. Unlike the original hand-rolled code, all rules, checks, castling,
en passant and promotion are delegated to python-chess, so the king/queen sit on
their standard files.
"""

import os

# Hide pygame's support-prompt banner. Set before importing pygame here too, so
# spawned MCTS workers that import this module first don't print it.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import chess
import pygame

from .sound import SoundEffect

# Display geometry (unchanged from Main.py).
WIDTH = HEIGHT = 720
SQNUM = 8
SQSIZE = WIDTH // SQNUM  # 90

# Move-history side panel sitting to the RIGHT of the 720x720 board.
PANEL_WIDTH = 340
TOTAL_WIDTH = WIDTH + PANEL_WIDTH  # full window width (board + panel)

# Colours (borrowed from Main.py's palette).
LIGHT = (238, 238, 210)
DARK = (118, 150, 86)
SELCOLOR = (186, 202, 68)     # selected source square
POSSCOLOR = (214, 214, 189)   # legal target squares
CHECKCOLOR = (235, 97, 80)    # king in check
TRACECOLOR = (246, 246, 105)  # last move's from/to squares
TRACEALPHA = 140              # how strongly the trace tints its squares

# Game-over "New Game" button.
NEWGAME_BG = (238, 238, 210)
NEWGAME_FG = (33, 31, 29)
NEWGAME_W, NEWGAME_H = 220, 64

# Promotion picker (the four choices offered when a pawn reaches the last rank).
# Offered top-to-bottom in this order; the human's promotion square is always
# row 0 (the far rank sits at the top in both orientations), so the picker is
# stacked downward from it and always stays on-board.
PROMO_ORDER = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
PROMO_CELL_BG = (245, 245, 245)
PROMO_CELL_BORDER = (33, 31, 29)

# Hint feature ("?" button + suggestion popup).
HINTCOLOR = (120, 190, 245)       # tint of the suggested move / popup accent
HINTALPHA = 150
HINT_BTN_BG = (120, 190, 245)
HINT_BTN_FG = (20, 30, 45)
HINT_POPUP_BG = (28, 26, 24)
HINT_POPUP_FG = (235, 235, 235)

# Move-log panel palette.
PANEL_BG = (33, 31, 29)
PANEL_TITLE = (240, 240, 240)
PANEL_WHITE_FG = (238, 238, 210)  # White's move lines
PANEL_BLACK_FG = (150, 190, 235)  # Black's move lines


class Board:
    """python-chess game state + the pygame rendering of it."""

    def __init__(self, screen, images, sound=None, chess_board=None,
                 flipped=False):
        self.screen = screen
        self.images = images                 # {piece.symbol(): Surface}
        self.sound = sound or SoundEffect()  # effect loader/player
        self.board = chess_board or chess.Board()
        self.selected = None                 # currently selected square, or None
        self.last_move = None                # last move played, for the trace
        self.new_game_rect = None            # game-over "New Game" button rect
        # Promotion picker: None, or {"to": square, "color": bool} while the
        # human is choosing which piece a promoting pawn becomes.
        self.promotion = None
        # Hint state: None, or {"move", "win_rate", "attackers"} — the current
        # MCTS suggestion to display. self.hint_rect is the "?" button rect.
        self.hint = None
        self.hint_rect = None
        # Human-readable log of every move played, for the side panel. Each
        # entry is (color, "[Player] *** Move Pawn from A to B").
        self.move_log = []
        self._fonts = None                   # lazily-built (big, small) fonts
        self._log_font = None                # lazily-built move-log font
        self._hint_font = None               # lazily-built (header, body) fonts
        # When flipped, the rank axis is mirrored so White sits at the bottom
        # (the human plays White). See the module docstring.
        self.flipped = flipped

    # ----- coordinate conversion (pixel <-> python-chess square) -----

    def square_at(self, pos):
        """Snap a mouse pixel position to a chess square, or None if off-board."""
        x, y = pos
        col, row = x // SQSIZE, y // SQSIZE
        if 0 <= col < SQNUM and 0 <= row < SQNUM:
            # Not flipped: row 0 == rank index 0 (White at top). Flipped mirrors
            # the rank so White ends up at the bottom.
            rank = (SQNUM - 1 - row) if self.flipped else row
            return chess.square(col, rank)
        return None

    def pixel_of(self, square):
        """Top-left pixel of a chess square."""
        col = chess.square_file(square)
        rank = chess.square_rank(square)
        row = (SQNUM - 1 - rank) if self.flipped else rank
        return col * SQSIZE, row * SQSIZE

    # ----- move helpers -----

    def target_squares(self, square):
        """Legal destination squares for the piece on `square`."""
        return [m.to_square for m in self.board.legal_moves
                if m.from_square == square]

    def push(self, move):
        """Play the matching sound and apply the move to the board state."""
        b = self.board
        # Describe the move for the log *before* pushing, while the moving piece
        # is still on its origin square.
        self.move_log.append((b.turn, self._describe_move(b, move)))
        # Classify before the move; check can only be judged afterwards.
        name = SoundEffect.classify(b, move)
        b.push(move)
        if b.is_check():
            name = "Check"
        self.sound.play(name)
        self.last_move = move
        self.selected = None
        # A move ends any open promotion pick and stales any shown hint.
        self.promotion = None
        self.hint = None

    @staticmethod
    def _describe_move(board, move):
        """Format a move as "[Player] *** Move Pawn from A to B".

        Read before the move is applied, so the piece still sits on
        ``move.from_square``.
        """
        player = "White" if board.turn == chess.WHITE else "Black"
        piece = board.piece_at(move.from_square)
        name = chess.piece_name(piece.piece_type).title() if piece else "Piece"
        frm = chess.square_name(move.from_square).upper()
        to = chess.square_name(move.to_square).upper()
        return f"[{player}] *** Move {name} from {frm} to {to}"

    # ----- rendering -----

    def draw(self):
        self._draw_squares()
        self._draw_highlights()
        if self.hint is not None:
            self._draw_hint_move()
        self._draw_pieces()
        if self.promotion is not None:
            self._draw_promotion()
        if self.board.is_game_over():
            self._draw_game_over()
        self._draw_move_log()
        self._draw_hint_button()
        if self.hint is not None and not self.board.is_game_over():
            self._draw_hint_popup()

    def _draw_squares(self):
        for row in range(SQNUM):
            for col in range(SQNUM):
                color = LIGHT if (row + col) % 2 == 0 else DARK
                pygame.draw.rect(
                    self.screen, color,
                    (col * SQSIZE, row * SQSIZE, SQSIZE, SQSIZE))

    def _draw_highlights(self):
        # Trace of the last move: tint its from and to squares (drawn first so
        # an active selection still shows on top).
        if self.last_move is not None:
            tint = pygame.Surface((SQSIZE, SQSIZE))
            tint.set_alpha(TRACEALPHA)
            tint.fill(TRACECOLOR)
            for sq in (self.last_move.from_square, self.last_move.to_square):
                self.screen.blit(tint, self.pixel_of(sq))
        # Selected square and its legal targets.
        if self.selected is not None:
            x, y = self.pixel_of(self.selected)
            pygame.draw.rect(self.screen, SELCOLOR, (x, y, SQSIZE, SQSIZE))
            for sq in self.target_squares(self.selected):
                tx, ty = self.pixel_of(sq)
                pygame.draw.rect(self.screen, POSSCOLOR,
                                 (tx, ty, SQSIZE, SQSIZE))
        # King in check.
        if self.board.is_check():
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                kx, ky = self.pixel_of(king_sq)
                pygame.draw.rect(self.screen, CHECKCOLOR,
                                 (kx, ky, SQSIZE, SQSIZE))

    def _draw_pieces(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue
            image = self.images.get(piece.symbol())
            if image is not None:
                self.screen.blit(image, self.pixel_of(square))

    def _get_fonts(self):
        """Return (big, small) fonts, building them once; None if unavailable."""
        if self._fonts is None:
            try:
                if not pygame.font.get_init():
                    pygame.font.init()
                self._fonts = (
                    pygame.font.SysFont("Arial", 72, bold=True),
                    pygame.font.SysFont("Arial", 30),
                )
            except Exception:
                self._fonts = (None, None)
        return self._fonts

    def _result_text(self):
        """Human-readable outcome, e.g. ('Game Over', 'Checkmate — Black wins')."""
        outcome = self.board.outcome()
        if outcome is None:
            return "Game Over", ""
        reason = outcome.termination.name.replace("_", " ").title()
        if outcome.winner is None:
            return "Game Over", f"{reason} — Draw"
        winner = "White" if outcome.winner == chess.WHITE else "Black"
        return "Game Over", f"{reason} — {winner} wins"

    def _draw_game_over(self):
        """Dim the board, print the result, and draw a "New Game" button."""
        big, small = self._get_fonts()
        # Dim overlay.
        veil = pygame.Surface((WIDTH, HEIGHT))
        veil.set_alpha(150)
        veil.fill((0, 0, 0))
        self.screen.blit(veil, (0, 0))
        # New Game button — drawn (and its rect stored) even without fonts so it
        # stays clickable in headless runs; the click is handled by Main.
        btn = pygame.Rect(0, 0, NEWGAME_W, NEWGAME_H)
        btn.center = (WIDTH // 2, HEIGHT // 2 + 120)
        self.new_game_rect = btn
        pygame.draw.rect(self.screen, NEWGAME_BG, btn, border_radius=10)
        if big is None:
            return
        if small is not None:
            label = small.render("New Game", True, NEWGAME_FG)
            self.screen.blit(label, label.get_rect(center=btn.center))
        title, subtitle = self._result_text()
        title_surf = big.render(title, True, (255, 255, 255))
        self.screen.blit(
            title_surf,
            title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 24)))
        if subtitle:
            sub_surf = small.render(subtitle, True, (220, 220, 220))
            self.screen.blit(
                sub_surf,
                sub_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40)))

    def new_game_clicked(self, pos):
        """True if `pos` hits the game-over "New Game" button."""
        return (self.board.is_game_over()
                and self.new_game_rect is not None
                and self.new_game_rect.collidepoint(pos))

    # ----- promotion picker -----

    def open_promotion(self, to_square, color):
        """Open the promotion picker for a pawn of `color` landing on `to_square`.

        Driven by HumanPlayer: while it is open, clicks are routed to
        ``promotion_choice_at`` instead of forming new moves.
        """
        self.promotion = {"to": to_square, "color": color}

    def _promotion_rects(self):
        """[(piece_type, Rect)] for the open picker, stacked below `to`."""
        x, _y = self.pixel_of(self.promotion["to"])
        return [(pt, pygame.Rect(x, i * SQSIZE, SQSIZE, SQSIZE))
                for i, pt in enumerate(PROMO_ORDER)]

    def _draw_promotion(self):
        """Draw the four promotion choices as a column from the target square."""
        color = self.promotion["color"]
        for pt, rect in self._promotion_rects():
            pygame.draw.rect(self.screen, PROMO_CELL_BG, rect)
            pygame.draw.rect(self.screen, PROMO_CELL_BORDER, rect, 2)
            image = self.images.get(chess.Piece(pt, color).symbol())
            if image is not None:
                self.screen.blit(image, rect.topleft)

    def promotion_choice_at(self, pos):
        """Piece type the click at `pos` selects in the picker, or None (cancel)."""
        if self.promotion is None:
            return None
        for pt, rect in self._promotion_rects():
            if rect.collidepoint(pos):
                return pt
        return None

    # ----- hint ("?") -----

    def hint_clicked(self, pos):
        """True if `pos` hits the "?" hint button (and the game is still on)."""
        return (self.hint_rect is not None
                and not self.board.is_game_over()
                and self.hint_rect.collidepoint(pos))

    def hint_attackers(self, move):
        """Names of enemy pieces that could recapture on `move`'s destination.

        Computed on the position *after* `move` is played (so it reflects the
        real recapture threat), from the mover's opponent. Duplicates (e.g. two
        pawns) are collapsed to one name. A hint aid, so pins are counted and
        en-passant recaptures ignored.
        """
        mover = self.board.turn
        after = self.board.copy()
        after.push(move)
        names = []
        for sq in after.attackers(not mover, move.to_square):
            piece = after.piece_at(sq)
            if piece is not None:
                name = chess.piece_name(piece.piece_type).title()
                if name not in names:
                    names.append(name)
        return names

    def _get_hint_font(self):
        """Return the (header, body) hint fonts, building them once."""
        if self._hint_font is None:
            try:
                if not pygame.font.get_init():
                    pygame.font.init()
                self._hint_font = (
                    pygame.font.SysFont("Arial", 26, bold=True),
                    pygame.font.SysFont("Arial", 20),
                )
            except Exception:
                self._hint_font = (None, None)
        return self._hint_font

    def _draw_hint_button(self):
        """Draw the "?" hint button in the panel's top-right corner."""
        btn = pygame.Rect(TOTAL_WIDTH - 16 - 46, 14, 46, 46)
        self.hint_rect = btn
        pygame.draw.rect(self.screen, HINT_BTN_BG, btn, border_radius=8)
        header, _body = self._get_hint_font()
        if header is not None:
            q = header.render("?", True, HINT_BTN_FG)
            self.screen.blit(q, q.get_rect(center=btn.center))

    def _draw_hint_move(self):
        """Tint the suggested move's from/to squares with the hint accent."""
        tint = pygame.Surface((SQSIZE, SQSIZE))
        tint.set_alpha(HINTALPHA)
        tint.fill(HINTCOLOR)
        move = self.hint["move"]
        for sq in (move.from_square, move.to_square):
            self.screen.blit(tint, self.pixel_of(sq))

    def _draw_hint_popup(self):
        """Draw the suggestion box: recommended move, win rate, recapture threats."""
        move = self.hint["move"]
        pct = round(self.hint["win_rate"] * 100)
        try:
            san = self.board.san(move)
        except Exception:
            san = move.uci()
        attackers = self.hint["attackers"]
        threat = ", ".join(attackers) if attackers else "none"
        lines = [
            f"Play {san}  —  {pct}% win chance",
            f"Can be recaptured by: {threat}",
        ]

        box = pygame.Rect(0, 0, 470, 118)
        box.center = (WIDTH // 2, 78)
        veil = pygame.Surface((box.width, box.height))
        veil.set_alpha(225)
        veil.fill(HINT_POPUP_BG)
        self.screen.blit(veil, box.topleft)
        pygame.draw.rect(self.screen, HINTCOLOR, box, 2, border_radius=8)

        header, body = self._get_hint_font()
        if header is None:
            return
        self.screen.blit(header.render("Hint", True, HINTCOLOR),
                         (box.x + 16, box.y + 10))
        for i, text in enumerate(lines):
            self.screen.blit(body.render(text, True, HINT_POPUP_FG),
                             (box.x + 16, box.y + 46 + i * 28))

    def _get_log_font(self):
        """Return the move-log (title, line) fonts, building them once."""
        if self._log_font is None:
            try:
                if not pygame.font.get_init():
                    pygame.font.init()
                self._log_font = (
                    pygame.font.SysFont("Arial", 24, bold=True),
                    pygame.font.SysFont("Consolas", 14),
                )
            except Exception:
                self._log_font = (None, None)
        return self._log_font

    @staticmethod
    def _wrap_text(text, font, max_width):
        """Break `text` into lines no wider than `max_width` pixels.

        Wraps on spaces; a single word wider than the panel is kept whole
        (rather than dropped) and simply overflows.
        """
        lines = []
        cur = ""
        for word in text.split(" "):
            trial = word if not cur else f"{cur} {word}"
            if not cur or font.size(trial)[0] <= max_width:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def _draw_move_log(self):
        """Draw the move-history panel to the right of the board.

        Shows the most recent moves that fit, newest at the bottom, each line
        reading ``[Player] *** Move Pawn from A to B`` (White and Black tinted
        differently). Lines too long for the panel are word-wrapped, with
        continuation lines indented. Silently draws just the panel background
        where fonts are unavailable (headless runs).
        """
        # Panel background.
        pygame.draw.rect(self.screen, PANEL_BG,
                         (WIDTH, 0, PANEL_WIDTH, HEIGHT))
        title_font, line_font = self._get_log_font()
        if title_font is None:
            return

        pad = 16
        x = WIDTH + pad
        title = title_font.render("Move History", True, PANEL_TITLE)
        self.screen.blit(title, (x, pad))

        line_h = line_font.get_linesize()
        top = pad + title.get_height() + 10
        avail = PANEL_WIDTH - 2 * pad
        # Flatten every entry into (colour, wrapped visual line); continuation
        # lines are indented so wrapped moves read as one entry.
        visual = []
        for color, text in self.move_log:
            for j, seg in enumerate(self._wrap_text(text, line_font, avail)):
                visual.append((color, seg if j == 0 else f"    {seg}"))
        # Keep only the most recent lines that fit, newest at the bottom.
        max_lines = max(1, (HEIGHT - top - pad) // line_h)
        for i, (color, seg) in enumerate(visual[-max_lines:]):
            fg = PANEL_WHITE_FG if color == chess.WHITE else PANEL_BLACK_FG
            surf = line_font.render(seg, True, fg)
            self.screen.blit(surf, (x, top + i * line_h))
