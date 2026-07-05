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


DEFAULT_EVALUATOR = "material"

# Name -> callable(board) -> [0, 1]. Register new ones with register_evaluator.
EVALUATORS = {DEFAULT_EVALUATOR: _material_evaluate}


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


def _search_visits(root_board, num_simulation, evaluator=DEFAULT_EVALUATOR):
    """Grow one MCTS tree and return ``{move.uci(): visits}`` for root children.

    ``evaluator`` is the *name* of the leaf evaluator; it is resolved to its
    callable once here (not per rollout).
    """
    evaluate = _get_evaluator(evaluator)
    root = _Node(root_board)
    for _ in range(num_simulation):
        node = _select(root)
        score = _rollout(node.board, evaluate)
        _backpropagate(node, score)
    return {child.move.uci(): child.visits for child in root.children}


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
