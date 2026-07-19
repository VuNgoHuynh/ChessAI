"""EnginePlayer entity: the AI player (plays whichever colour it is given).

Move selection is a Monte Carlo Tree Search (MCTS). Each call to
``select_move`` runs playouts from the current position and returns the root
move that was visited most often.

Each simulation walks four phases:

* **Selection** — descend the tree by UCT until reaching a node with untried
  moves (or a terminal position).
* **Expansion** — add one child for an untried move.
* **Rollout** — play random moves from the new node up to a ply cap, then score
  the leaf position with the configured *evaluator* (material by default).
* **Backpropagation** — push the score back up, crediting each node from the
  perspective of the side that moved into it.

**Pluggable evaluator.** The one place a stronger model plugs in is the leaf
evaluator (see :func:`register_evaluator`): a callable ``board -> float`` in
``[0, 1]`` from White's perspective. Evaluators are referenced by *name*, so
only a string crosses the process boundary to spawn workers — each worker
resolves the name against its own registry. The search itself is unchanged; a
future neural network becomes just another registered evaluator.

**Root parallelization.** The search is embarrassingly parallel: ``num_workers``
independent trees are grown in separate processes, each running the full
``num_simulation`` budget, then their per-root-move visit counts are summed and
the most-visited move wins. This multiplies the effective simulation count by
``num_workers`` in roughly the same wall-clock time as a single serial tree.
Because python-chess rollouts are pure-Python and CPU-bound (the GIL rules out
threads), we use ``multiprocessing``. The pool is created lazily and reused
across moves; any failure (or ``num_workers <= 1``) falls back to a serial
in-process search.
"""

import atexit
import math
import multiprocessing
import os
import random

import chess

# Rough piece values (centipawns) for the rollout cutoff evaluation.
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Exploration constant for UCT (~sqrt(2)).
UCT_C = 1.41

# How many plies a rollout plays before falling back to a material estimate.
ROLLOUT_DEPTH = 40


class _Node:
    """A search-tree node wrapping one python-chess position."""

    __slots__ = ("board", "parent", "move", "children", "untried", "visits", "wins")

    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move                       # move that led here (None at root)
        self.children = []
        self.untried = list(board.legal_moves)
        self.visits = 0
        # Total score from the perspective of the side that moved into this node.
        self.wins = 0.0

    def is_fully_expanded(self):
        return not self.untried

    def best_child(self, c=UCT_C):
        """Pick the child maximising the UCT score."""
        log_n = math.log(self.visits)
        best, best_score = None, -1.0
        for child in self.children:
            exploit = child.wins / child.visits
            explore = c * math.sqrt(log_n / child.visits)
            score = exploit + explore
            if score > best_score:
                best, best_score = child, score
        return best


# ----- MCTS phases (module-level so worker processes can run them) -----
#
# None of these touch engine state: the search is colour-agnostic and
# ``_evaluate`` scores from White's perspective in [0, 1].

def _select(node):
    """Descend to a node to expand: follow UCT, expanding the first chance."""
    while not node.board.is_game_over():
        if not node.is_fully_expanded():
            return _expand(node)
        node = node.best_child()
    return node


def _expand(node):
    """Add and return one child for an untried move of `node`."""
    move = node.untried.pop()
    child_board = node.board.copy()
    child_board.push(move)
    child = _Node(child_board, parent=node, move=move)
    node.children.append(child)
    return child


def _rollout(board, evaluate):
    """Play random moves up to a depth cap; return a White-perspective score.

    ``evaluate`` is the leaf evaluator (see :func:`register_evaluator`). The
    returned score is in [0, 1]: 1.0 is winning for White, 0.0 for Black, 0.5
    an even/drawn position.
    """
    sim = board.copy()
    for _ in range(ROLLOUT_DEPTH):
        if sim.is_game_over():
            break
        sim.push(random.choice(list(sim.legal_moves)))
    return evaluate(sim)


# ----- Pluggable position evaluators -----
#
# An evaluator is a callable ``board -> float`` scoring the position for White
# in [0, 1]. This is the single seam for a "better model": swap the material
# heuristic below for a neural network without touching the search. Evaluators
# are keyed by NAME (see EVALUATORS / register_evaluator) so only the name — not
# the callable — has to cross the process boundary to spawn workers, which each
# resolve it against their own registry.

def _material_evaluate(board):
    """Default evaluator: material balance squashed through a logistic curve."""
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return 0.0
        return 0.5
    diff = 0
    for piece_type, value in PIECE_VALUE.items():
        diff += value * len(board.pieces(piece_type, chess.WHITE))
        diff -= value * len(board.pieces(piece_type, chess.BLACK))
    return 1.0 / (1.0 + math.exp(-diff / 400.0))


# Piece-square tables (centipawns), one per piece type, 64 entries each in
# the standard "a8 first" layout used throughout chess literature: index 0 is
# a8, index 7 is h8, ... index 56 is a1, index 63 is h1 (row-major, top rank
# first). Values encode positional preferences from White's point of view
# (e.g. pawns are worth more the closer they get to promotion on rank 8).
_PAWN_PST = (
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_KNIGHT_PST = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)
_BISHOP_PST = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)
_ROOK_PST = (
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
)
_QUEEN_PST = (
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
)
_KING_PST = (
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
)

# piece type -> its table above.
PST = {
    chess.PAWN: _PAWN_PST,
    chess.KNIGHT: _KNIGHT_PST,
    chess.BISHOP: _BISHOP_PST,
    chess.ROOK: _ROOK_PST,
    chess.QUEEN: _QUEEN_PST,
    chess.KING: _KING_PST,
}


def _pst_value(table, square, color):
    """Look up `table` for `square`, from `color`'s point of view.

    The tables above are written a8-first, i.e. indexed by
    ``chess.square_mirror(square)`` for a White piece (mirroring the rank
    turns python-chess's a1-based numbering into the table's a8-based
    reading order). A Black piece on `square` is worth what a White piece
    would be worth on the vertically-mirrored square, so it indexes the same
    table directly with `square` (no mirroring) — the double mirror cancels.
    """
    return table[chess.square_mirror(square) if color == chess.WHITE else square]


def _pst_evaluate(board):
    """Evaluator: material balance plus piece-square positional bonuses.

    Same White-perspective [0, 1] contract and logistic squash as
    :func:`_material_evaluate`, just with a richer centipawn `diff` that also
    rewards well-placed pieces, not just their raw count.
    """
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return 0.0
        return 0.5
    diff = 0
    for piece_type, value in PIECE_VALUE.items():
        table = PST[piece_type]
        for square in board.pieces(piece_type, chess.WHITE):
            diff += value + _pst_value(table, square, chess.WHITE)
        for square in board.pieces(piece_type, chess.BLACK):
            diff -= value + _pst_value(table, square, chess.BLACK)
    return 1.0 / (1.0 + math.exp(-diff / 400.0))


DEFAULT_EVALUATOR = "material"

# Name -> callable(board) -> [0, 1]. Register new ones with register_evaluator.
EVALUATORS = {DEFAULT_EVALUATOR: _material_evaluate}
# Import-time registration: see register_evaluator's docstring below for why
# this must happen here rather than under `if __name__ == "__main__"`.
EVALUATORS["pst"] = _pst_evaluate


def register_evaluator(name, fn):
    """Register a position evaluator ``fn`` (``board -> [0, 1]``) under ``name``.

    Call this at *import time* (module top level), not inside ``if __name__ ==
    '__main__'``: spawn worker processes re-import their entry module, so an
    import-time registration is visible to them and resolvable by name. A
    stateful/GPU model is best constructed once per worker via a pool
    ``initializer`` rather than rebuilt per rollout.
    """
    EVALUATORS[name] = fn


def _get_evaluator(name):
    """Resolve an evaluator name to its callable, or raise a clear KeyError."""
    try:
        return EVALUATORS[name]
    except KeyError:
        raise KeyError(
            f"Unknown evaluator {name!r}; registered: {sorted(EVALUATORS)}")


def _backpropagate(node, white_score):
    """Credit the playout up the tree, flipping perspective per node."""
    while node is not None:
        node.visits += 1
        # The side that moved into `node` is the opponent of the side to move.
        mover = not node.board.turn
        node.wins += white_score if mover == chess.WHITE else 1.0 - white_score
        node = node.parent


def _search_stats(root_board, num_simulation, evaluator=DEFAULT_EVALUATOR):
    """Grow one MCTS tree and return ``{move.uci(): (visits, wins)}`` per child.

    This is the shared search primitive. ``wins`` for a root child is the total
    score credited from the perspective of the side that *moved into* the child
    — i.e. the side to move at the root (the mover) — so ``wins / visits`` is
    already that mover's win probability for the move, for either colour.

    ``evaluator`` is the *name* of the leaf evaluator; it is resolved to its
    callable once here (not per rollout).
    """
    evaluate = _get_evaluator(evaluator)
    root = _Node(root_board)
    for _ in range(num_simulation):
        node = _select(root)
        score = _rollout(node.board, evaluate)
        _backpropagate(node, score)
    return {child.move.uci(): (child.visits, child.wins)
            for child in root.children}


def _search_visits(root_board, num_simulation, evaluator=DEFAULT_EVALUATOR):
    """Grow one MCTS tree and return ``{move.uci(): visits}`` for root children.

    Thin wrapper over :func:`_search_stats` (the single tree driver) that drops
    the win totals, used by the move-selection play path.
    """
    return {uci: visits for uci, (visits, _wins)
            in _search_stats(root_board, num_simulation, evaluator).items()}


def _worker(args):
    """Pool task: run one independent tree from a FEN, returning root visits.

    Reseeds from fresh OS entropy so each task (a persistent pool runs many)
    explores a distinct random stream. The evaluator name is resolved inside
    the worker against this module's registry. ``chess960`` must match the root
    board's variant so castling in Fischer-random positions is generated
    correctly (a Chess960 FEN parsed in standard mode silently drops castling).
    """
    fen, num_simulation, evaluator, chess960 = args
    random.seed()
    board = chess.Board(fen, chess960=chess960)
    return _search_visits(board, num_simulation, evaluator)


def _worker_stats(args):
    """Pool task like :func:`_worker` but returning ``{uci: (visits, wins)}``.

    Used by the hint search (``EnginePlayer.suggest``) so per-move win rates
    can be aggregated across the independent trees. Import- and spawn-safe (a
    module-level function) exactly like ``_worker``.
    """
    fen, num_simulation, evaluator, chess960 = args
    random.seed()
    board = chess.Board(fen, chess960=chess960)
    return _search_stats(board, num_simulation, evaluator)


class EnginePlayer:
    """MCTS-based chess engine. Plays a fixed colour (White by default).

    ``num_simulation`` is the playout budget *per worker*; with ``num_workers``
    processes the effective budget is ``num_simulation * num_workers``.
    ``num_workers`` defaults to the CPU count; pass ``1`` to force the serial
    in-process search.

    ``evaluator`` names the leaf position evaluator (see
    :func:`register_evaluator`); it defaults to the built-in material heuristic
    and is where a stronger model is dropped in.

    The external contract other code depends on is just ``select_move(board)``
    (returning a ``chess.Move`` or ``None``) and ``color`` — a drop-in
    replacement engine only needs those two.
    """

    def __init__(self, color=chess.WHITE, num_simulation=500, num_workers=None,
                 evaluator=DEFAULT_EVALUATOR):
        self.color = color
        self.num_simulation = num_simulation
        _get_evaluator(evaluator)  # fail fast on an unknown name
        self.evaluator = evaluator
        if num_workers is None:
            num_workers = os.cpu_count() or 1
        self.num_workers = max(1, num_workers)
        self._pool = None  # created lazily on first parallel search

    def _get_pool(self):
        """Lazily create (and register for cleanup) the process pool."""
        if self._pool is None:
            self._pool = multiprocessing.Pool(processes=self.num_workers)
            atexit.register(self._close_pool)
        return self._pool

    def _close_pool(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def select_move(self, board):
        """Return the chosen chess.Move for `board` (an entity.Board), or None.

        None is returned only when there is nothing to play (game over or no
        legal moves). Otherwise MCTS always yields a legal move.
        """
        state = board.board
        if state.is_game_over():
            return None
        legal = list(state.legal_moves)
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        if self.num_workers > 1:
            uci = self._parallel_best_uci(state)
            if uci is not None:
                return chess.Move.from_uci(uci)
            # Parallel path failed; fall through to the serial search.

        visits = _search_visits(state.copy(), self.num_simulation, self.evaluator)
        return chess.Move.from_uci(max(visits, key=visits.get))

    def suggest(self, board):
        """Return an MCTS hint for the side to move on ``board`` (entity.Board).

        Colour-agnostic: it evaluates whoever is to move — used to hint the
        *human*, independently of ``self.color`` (the search never reads it).
        Returns a dict, or ``None`` when there is nothing to play (game over /
        no legal moves)::

            {
                "move":      chess.Move,   # the most-visited (recommended) move
                "win_rate":  float,        # that move's win probability, [0, 1]
                "win_rates": {uci: float}, # every explored move's win rate
            }

        ``win_rate`` is the mover's own win probability: a root child's
        ``wins / visits`` is credited from the mover's perspective by
        :func:`_backpropagate`, so it is reported directly (no re-derivation)
        for either colour. The recommended move is the most-*visited* one (the
        robust MCTS choice); its win rate is displayed alongside for
        consistency.
        """
        state = board.board
        if state.is_game_over():
            return None
        if not any(state.legal_moves):
            return None

        stats = None
        if self.num_workers > 1:
            stats = self._parallel_stats(state)
        if stats is None:  # serial in-process fallback
            stats = _search_stats(state.copy(), self.num_simulation,
                                  self.evaluator)
        if not stats:
            return None

        win_rates = {uci: (wins / visits if visits else 0.0)
                     for uci, (visits, wins) in stats.items()}
        best_uci = max(stats, key=lambda u: stats[u][0])  # most visited
        return {
            "move": chess.Move.from_uci(best_uci),
            "win_rate": win_rates[best_uci],
            "win_rates": win_rates,
        }

    def _parallel_stats(self, state):
        """Run ``num_workers`` trees in parallel; return summed per-move stats.

        Returns ``{uci: (visits, wins)}`` aggregated across the independent
        trees, or ``None`` on any multiprocessing failure so ``suggest`` can
        fall back to a serial search.
        """
        fen = state.fen()
        tasks = [(fen, self.num_simulation, self.evaluator, state.chess960)] * self.num_workers
        try:
            results = self._get_pool().map(_worker_stats, tasks)
        except Exception:
            self._close_pool()
            return None

        totals = {}
        for result in results:
            for uci, (visits, wins) in result.items():
                v, w = totals.get(uci, (0, 0.0))
                totals[uci] = (v + visits, w + wins)
        return totals or None

    def _parallel_best_uci(self, state):
        """Run ``num_workers`` trees in parallel; return the winning move's UCI.

        Returns ``None`` on any multiprocessing failure so the caller can fall
        back to a serial search (e.g. sandboxed environments that forbid
        spawning processes).
        """
        fen = state.fen()
        tasks = [(fen, self.num_simulation, self.evaluator, state.chess960)] * self.num_workers
        try:
            results = self._get_pool().map(_worker, tasks)
        except Exception:
            self._close_pool()
            return None

        # Sum visit counts for each move across every independent tree.
        totals = {}
        for result in results:
            for uci, visits in result.items():
                totals[uci] = totals.get(uci, 0) + visits
        if not totals:
            return None
        return max(totals, key=totals.get)
