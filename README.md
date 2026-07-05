# ChessAI ♟️

A chess game built with **pygame** and **python-chess**, where you play against a **Monte Carlo Tree Search (MCTS)** engine. Pick your side, pick your mode, and play a full game with legal-move highlighting, check detection, sound effects, and a live move-history panel — all rendered with the board always oriented so your pieces sit at the bottom.

If you like this project, **consider giving it a ⭐ — it helps others find it!**

## Features

- **Full chess rules** via [python-chess](https://python-chess.readthedocs.io/) — legal moves, castling, en passant, promotion, checkmate/stalemate detection.
- **Monte Carlo Tree Search AI** — UCT selection, random rollouts, and backpropagation, parallelized across CPU cores via `multiprocessing` for a much larger effective search budget per move.
- **Pluggable leaf evaluator** — swap the position-scoring function used at the end of each rollout. Ships with a material evaluator and a piece-square-table (PST) evaluator; adding your own (e.g. a neural net) is a one-function drop-in.
- **Two start menus** — choose your side (White or Black), then your mode:
  - **Classic** — the standard starting position.
  - **Random** — a real mid-game tactical puzzle pulled live from the [Lichess puzzle API](https://lichess.org/api/puzzle/next) (falls back to a random Chess960/Fischer-random start if offline).
- **Polished board UI** — last-move highlighting, selected-piece and legal-target squares, king-in-check highlighting, a move-history side panel, and a "New Game" button on the game-over screen.
- **Sound effects** for moves, captures, castling, checks, and promotions.

## Getting started

### Requirements

- Python 3.12
- [`pygame`](https://www.pygame.org/)
- [`python-chess`](https://python-chess.readthedocs.io/) 1.10.0

### Run it

```bash
pip install pygame python-chess
python Main.py
```

Run from the repo root — the game loads assets from `Images/` and `SoundEffect/` using paths relative to the project.

### How to play

1. Choose your side — White or Black.
2. Choose your mode — Classic (standard start) or Random (a live Lichess puzzle).
3. Click a piece to select it, then click a highlighted square to move. The AI replies automatically.

## How the AI works

The engine (`entity/engine_player.py`) runs an MCTS search per move:

1. **Select** — walk down the tree using UCT (Upper Confidence bound for Trees) to balance exploration and exploitation.
2. **Expand** — add one new child node for an untried move.
3. **Rollout** — play randomly to a capped depth, then score the resulting position with the configured **evaluator** (material balance or piece-square tables), from White's perspective.
4. **Backpropagate** — update visit counts and value estimates up the tree.

The search fans out across your CPU cores: each worker process runs its own independent tree, and the results are merged by summing visit counts per move — the most-visited move wins.

## Project structure

```
Main.py                 # game loop, start menus, engine/human turn handling
entity/
  board.py              # python-chess board wrapper + pygame rendering
  board_generator.py    # asset loading, board/puzzle construction
  human_player.py       # mouse-click -> move conversion
  engine_player.py      # MCTS engine, evaluators, multiprocessing search
  sound.py              # sound effect loading and playback
Images/                 # piece sprites
SoundEffect/            # move/capture/castle/check/promote clips
```

## Contributing

Issues and pull requests are welcome — whether it's a bug fix, a UI polish, or a new leaf evaluator for the MCTS engine.

## License

[MIT](LICENSE)
