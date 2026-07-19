"""Chess AI — pygame front end.

A start menu lets the player pick their side; the human plays that colour and the
AI plays the other. The board is oriented so the human's pieces always sit at the
BOTTOM of the screen (so it flips when the human plays White). All board state and
move legality live in python-chess, wrapped by ``entity.Board``. Mouse input is
turned into moves by ``entity.HumanPlayer``; the AI's moves come from
``entity.EnginePlayer``.
"""

import os
import random

# Hide pygame's "Hello from the pygame community" banner. Set before importing
# pygame, and here (module top) so spawned MCTS workers — which re-import this
# module on Windows — stay quiet too, instead of each printing the banner.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import chess
import pygame

from entity import BoardGenerator, EnginePlayer, HumanPlayer
from entity.board import HEIGHT, TOTAL_WIDTH, WIDTH

# How long the AI "thinks" before playing, in milliseconds.
ENGINE_DELAY = 300

# --- Start-menu palette ---
MENU_BG = (49, 46, 43)
MENU_TITLE = (240, 240, 240)
BTN_WHITE_BG = (238, 238, 210)
BTN_WHITE_FG = (49, 46, 43)
BTN_BLACK_BG = (60, 64, 72)
BTN_BLACK_FG = (240, 240, 240)
# Mode-menu buttons (Classic / Random).
BTN_CLASSIC_BG = (238, 238, 210)
BTN_CLASSIC_FG = (49, 46, 43)
BTN_RANDOM_BG = (150, 190, 235)
BTN_RANDOM_FG = (20, 30, 45)


def _button_menu(screen, title, buttons):
    """Draw a titled menu of buttons and block until one is clicked.

    ``buttons`` is a list of ``(bg, fg, label, value)``; the buttons are stacked
    vertically and centred. Returns the clicked button's ``value``, or ``None``
    if the window was closed first.
    """
    btn_w, btn_h, gap = 320, 90, 30
    cx = TOTAL_WIDTH // 2
    # Stack the buttons vertically, centred on the window below the title.
    block_h = len(buttons) * btn_h + (len(buttons) - 1) * gap
    top = (HEIGHT - block_h) // 2 + 40
    rects = [pygame.Rect(cx - btn_w // 2, top + i * (btn_h + gap), btn_w, btn_h)
             for i in range(len(buttons))]

    try:
        if not pygame.font.get_init():
            pygame.font.init()
        title_font = pygame.font.SysFont("Arial", 56, bold=True)
        btn_font = pygame.font.SysFont("Arial", 34, bold=True)
    except Exception:
        title_font = btn_font = None

    def render():
        screen.fill(MENU_BG)
        if title_font is not None:
            title_surf = title_font.render(title, True, MENU_TITLE)
            screen.blit(title_surf, title_surf.get_rect(center=(cx, top - 90)))
        for rect, (bg, fg, label, _value) in zip(rects, buttons):
            pygame.draw.rect(screen, bg, rect, border_radius=10)
            if btn_font is not None:
                text = btn_font.render(label, True, fg)
                screen.blit(text, text.get_rect(center=rect.center))
        pygame.display.flip()

    clock = pygame.time.Clock()
    while True:
        render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, (_bg, _fg, _label, value) in zip(rects, buttons):
                    if rect.collidepoint(event.pos):
                        return value
        clock.tick(60)


def _loading_screen(screen, text):
    """Paint a single centred-text frame (e.g. while a blocking fetch runs)."""
    screen.fill(MENU_BG)
    try:
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.SysFont("Arial", 44, bold=True)
        surf = font.render(text, True, MENU_TITLE)
        screen.blit(surf, surf.get_rect(center=(TOTAL_WIDTH // 2, HEIGHT // 2)))
    except Exception:
        pass
    pygame.display.flip()


def show_menu(screen):
    """Draw the side-selection menu (screen 1) and block until a side is picked.

    Returns ``chess.WHITE`` or ``chess.BLACK`` for the human's chosen side, or
    ``None`` if the window was closed before a choice was made.
    """
    return _button_menu(screen, "Choose Your Side", [
        (BTN_WHITE_BG, BTN_WHITE_FG, "Play as White", chess.WHITE),
        (BTN_BLACK_BG, BTN_BLACK_FG, "Play as Black", chess.BLACK),
    ])


def show_mode_menu(screen):
    """Draw the mode-selection menu (screen 2) and block until a mode is picked.

    Returns ``"classic"`` (standard start position) or ``"random"`` (a real
    mid-game puzzle fetched from Lichess), or ``None`` if the window was closed
    first.
    """
    return _button_menu(screen, "Choose Mode", [
        (BTN_CLASSIC_BG, BTN_CLASSIC_FG, "Classic", "classic"),
        (BTN_RANDOM_BG, BTN_RANDOM_FG, "Random", "random"),
    ])


def _new_board(screen, player_color, mode):
    """Build a fresh Board for the given side and mode.

    Random mode drops the player into a real mid-game puzzle fetched from
    Lichess (a standard-chess position, chess960=False). The fetch is a
    blocking network call, so show a "Loading" frame first; if it fails we
    fall back to a random Chess960 start (chess960=True) so the mode still
    works offline. Classic uses the standard start. The human always sits at
    the bottom, so the board is flipped when they play White.
    """
    fen, chess960 = None, False
    if mode == "random":
        _loading_screen(screen, "Loading puzzle...")
        fen = BoardGenerator.random_puzzle_fen()
        if fen is None:  # offline / API failure — keep the mode "random"
            fen = BoardGenerator.random_chess960_fen()
            chess960 = True
    return BoardGenerator.generate(
        screen, fen=fen, flipped=(player_color == chess.WHITE),
        chess960=chess960)


def main(max_frames=None, player_color=None, mode=None):
    """Run the game loop.

    ``player_color`` is the human's side (``chess.WHITE``/``chess.BLACK``); the AI
    takes the other colour. Left as ``None`` it is chosen via the start menu.
    ``mode`` is ``"classic"`` (standard start) or ``"random"`` (a mid-game puzzle
    fetched from Lichess); left as ``None`` it is chosen via the mode menu
    (screen 2), or defaults to ``"classic"`` for headless callers. Headless callers that pass
    ``max_frames`` default to Black (the original layout) so the menu never
    blocks. ``max_frames`` bounds the number of iterations for tests; ``None``
    runs until the window is closed.

    When the game ends, a "New Game" button appears on the game-over overlay;
    clicking it starts a fresh game with the same side and mode (a fresh puzzle
    in Random mode). ``max_frames`` is a total budget across such restarts.
    """
    pygame.init()
    screen = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
    pygame.display.set_caption("Chess AI")

    if player_color is None:
        if max_frames is not None:
            player_color = chess.BLACK
        else:
            player_color = show_menu(screen)
            if player_color is None:  # window closed at the menu
                pygame.quit()
                return None

    if mode is None:
        if max_frames is not None:
            mode = "classic"
        else:
            mode = show_mode_menu(screen)
            if mode is None:  # window closed at the mode menu
                pygame.quit()
                return None

    board = _new_board(screen, player_color, mode)
    human = HumanPlayer(player_color)
    engine = EnginePlayer(not player_color)
    clock = pygame.time.Clock()

    run = True
    frames = 0
    while run:
        board.draw()
        pygame.display.flip()

        if not board.board.is_game_over():
            if board.board.turn == engine.color:
                # AI to move.
                move = engine.select_move(board)
                if move is None:
                    # Defensive net: MCTS returns a move whenever one exists, so
                    # this only triggers in a genuinely move-less position.
                    legal = list(board.board.legal_moves)
                    move = random.choice(legal) if legal else None
                if move is not None:
                    pygame.time.wait(ENGINE_DELAY)
                    board.push(move)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if board.new_game_clicked(event.pos):
                    # Game over: restart a fresh game, same side and mode.
                    board = _new_board(screen, player_color, mode)
                elif not board.board.is_game_over() and board.board.turn == human.color:
                    move = human.handle_click(event.pos, board)
                    if move is not None:
                        board.push(move)

        clock.tick(60)
        frames += 1
        if max_frames is not None and frames >= max_frames:
            run = False

    pygame.quit()
    return board


if __name__ == "__main__":
    main()
