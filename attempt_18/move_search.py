"""
Shallow value-head search for move selection (with top-K policy pruning).

`choose_move(model, board, device, search_depth, temperature)` is the single entry
point used by both self-play generation (generate_sf_training_data.py) and the
vs-Stockfish evaluation (evaluate_vs_stockfish.py):

  * search_depth <= 0 -> pure policy-head move: legal-move-masked argmax of the
    policy logits, or a softmax sample over them when `temperature` is large.

  * search_depth == k >= 1 -> negamax `k` plies deep with **top-K policy
    pruning** at every interior node (root included).  At each level we run a
    single batched forward pass on the level's interior boards to read their
    policy heads, then expand the top-K legal replies per node **plus every
    forcing move** (captures, checks, promotions).  Forcing-move inclusion
    keeps tactics visible to the search even when the policy is weak - in
    particular, a mate-in-1 is always a check, so it can never be pruned.
    The leaf level is then scored in a single batched value-head forward pass
    and the values are negamaxed back to the root.

    With branching cap `_TOP_K = 8` and ~5 forcing moves per midgame node,
    depth=2 visits ~180-240 positions instead of ~35^2 ~= 1225 with full-
    width expansion - roughly 5-7x less compute, while staying tactically
    sound on captures/checks/promotions.

`temperature` controls the final pick: deterministic argmax when small,
otherwise softmax over the negamaxed root-move values (or policy logits for
search_depth <= 0).
"""

import chess
import numpy as np
import torch

from chess_engine import board_to_tensor, move_to_policy_index

# Sentinel just outside the value head's [-1, 1] range so a mate dominates
# any heuristic eval after sign-flipping through negamax.
_TERMINAL_WIN = 2.0
_ARGMAX_TEMP = 0.2

# Top-K branching cap at every interior node during k-ply search.  Tune via
# benchmark_play.py: bigger = more thorough, smaller = faster.  At depth=2
# this caps the leaf count at K^2 (so K=8 -> 64 leaves per move).
_TOP_K = 8


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _terminal_value_stm(board):
    """Game-over value from the side-to-move's POV, or None if not over.

    Uses claim_draw=True so a weak net's threefold/50-move shuffles count as
    draws, consistent with how the actual games are adjudicated.
    """
    if board.is_checkmate():
        return -_TERMINAL_WIN
    if board.is_game_over(claim_draw=True):
        return 0.0
    return None


def _sample_or_argmax(items, scores, temperature):
    scores = np.asarray(scores, dtype=np.float64)
    t = float(temperature) if temperature is not None else 0.0
    if t <= _ARGMAX_TEMP:
        return items[int(np.argmax(scores))]
    z = scores / t
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return items[int(np.random.choice(len(items), p=p))]


def _move_logit(flat_policy, mv):
    idx = move_to_policy_index(mv.uci())
    if idx is None:
        return -1e9
    src_row, src_col, plane = idx
    return float(flat_policy[plane * 64 + src_row * 8 + src_col])


def _amp_ctx(device):
    """Inference autocast context: fp16 on CUDA, no-op elsewhere.

    Half-precision roughly halves matmul/conv time on Ampere+ GPUs at no
    practical accuracy cost for inference; on CPU fp16 is generally slower
    than fp32, so we disable it there.
    """
    dev_type = device.type if isinstance(device, torch.device) else torch.device(device).type
    return torch.autocast(device_type=dev_type, dtype=torch.float16,
                          enabled=(dev_type == "cuda"))


@torch.no_grad()
def _value_white_pov(model, boards, device, batch_size=1024):
    """Batched value-head eval. Returns np.array of white-POV values in [-1, 1]."""
    if not boards:
        return np.zeros(0, dtype=np.float64)
    out = np.empty(len(boards), dtype=np.float64)
    for i in range(0, len(boards), batch_size):
        chunk = boards[i:i + batch_size]
        x = torch.stack([board_to_tensor(b) for b in chunk]).to(device)
        with _amp_ctx(device):
            value, _policy = model(x)
        out[i:i + len(chunk)] = value.reshape(-1).float().cpu().numpy()
    return out


@torch.no_grad()
def _policy_logits_batch(model, boards, device, batch_size=1024):
    """Batched policy-head eval. Returns np.array (B, 4672) of flat policy logits."""
    if not boards:
        return np.zeros((0, 4672), dtype=np.float64)
    out = np.empty((len(boards), 4672), dtype=np.float64)
    for i in range(0, len(boards), batch_size):
        chunk = boards[i:i + batch_size]
        x = torch.stack([board_to_tensor(b) for b in chunk]).to(device)
        with _amp_ctx(device):
            _value, policy = model(x)
        out[i:i + len(chunk)] = policy.reshape(len(chunk), -1).float().cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# policy-head move (search_depth <= 0)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _policy_logits(model, board, device):
    legal = list(board.legal_moves)
    x = board_to_tensor(board).unsqueeze(0).to(device)
    with _amp_ctx(device):
        _value, policy_logits = model(x)
    flat = policy_logits.reshape(-1).float().cpu().numpy()
    logits = np.full(len(legal), -1e9, dtype=np.float64)
    for i, mv in enumerate(legal):
        logits[i] = _move_logit(flat, mv)
    return legal, logits


def _policy_move(model, board, device, temperature):
    legal, logits = _policy_logits(model, board, device)
    return _sample_or_argmax(legal, logits, temperature)


# ---------------------------------------------------------------------------
# k-ply value search with top-K policy pruning
# ---------------------------------------------------------------------------

def _expand_moves(board, flat_policy, k):
    """Top-K policy moves unioned with all forcing moves (captures, checks,
    promotions).  Forcing-move inclusion keeps mate-in-1 / material-winning
    tactics visible even when the policy is weak."""
    legal = list(board.legal_moves)
    if len(legal) <= k:
        return legal
    logits = np.array([_move_logit(flat_policy, mv) for mv in legal])
    keep = set(int(i) for i in np.argpartition(-logits, k - 1)[:k])
    for i, mv in enumerate(legal):
        if mv.promotion is not None or board.is_capture(mv) or board.gives_check(mv):
            keep.add(i)
    return [legal[i] for i in keep]


def _search_move(model, board, device, depth, temperature, top_k=_TOP_K):
    legal = list(board.legal_moves)
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]

    # nodes[0] is root; each entry: {"board", "children": [(mv, cidx)], "value": float|None}
    nodes = [{"board": board, "children": [], "value": None}]
    frontier = [0]  # node indices at the current depth (level 0 = root)

    for _level in range(depth):
        # Mark terminal frontier nodes (except root - choose_move must still
        # return a legal move even if the root is draw-claimable).
        to_expand = []
        for nidx in frontier:
            tv = None if nidx == 0 else _terminal_value_stm(nodes[nidx]["board"])
            if tv is not None:
                nodes[nidx]["value"] = tv
            else:
                to_expand.append(nidx)

        if not to_expand:
            frontier = []
            break

        # One batched forward over this level's interior boards -> policy heads.
        boards = [nodes[nidx]["board"] for nidx in to_expand]
        pols = _policy_logits_batch(model, boards, device)

        next_frontier = []
        for i, nidx in enumerate(to_expand):
            nb = nodes[nidx]["board"]
            for mv in _expand_moves(nb, pols[i], top_k):
                cb = nb.copy()                                   # keeps move stack (repetition detection)
                cb.push(mv)
                cidx = len(nodes)
                nodes.append({"board": cb, "children": [], "value": None})
                nodes[nidx]["children"].append((mv, cidx))
                next_frontier.append(cidx)

        frontier = next_frontier

    # Score the leaves (everything left in the final frontier).
    leaves_to_score = []
    for nidx in frontier:
        tv = _terminal_value_stm(nodes[nidx]["board"])
        if tv is not None:
            nodes[nidx]["value"] = tv
        else:
            leaves_to_score.append(nidx)
    if leaves_to_score:
        leaf_boards = [nodes[nidx]["board"] for nidx in leaves_to_score]
        vals_w = _value_white_pov(model, leaf_boards, device)
        for nidx, vw in zip(leaves_to_score, vals_w):
            b = nodes[nidx]["board"]
            nodes[nidx]["value"] = float(vw) if b.turn == chess.WHITE else -float(vw)

    # Negamax back-up.
    def negamax(nidx):
        node = nodes[nidx]
        if not node["children"]:
            return node["value"]
        return max(-negamax(cidx) for _mv, cidx in node["children"])

    if not nodes[0]["children"]:
        return None
    move_scores = [(mv, -negamax(cidx)) for mv, cidx in nodes[0]["children"]]
    moves = [mv for mv, _s in move_scores]
    scores = [s for _mv, s in move_scores]
    return _sample_or_argmax(moves, scores, temperature)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def choose_move(model, board, device, search_depth=1, temperature=0.0):
    """Pick a move for `model` to play in `board`. See the module docstring."""
    if search_depth and int(search_depth) >= 1:
        mv = _search_move(model, board, device, int(search_depth), temperature)
        if mv is not None:
            return mv
    return _policy_move(model, board, device, temperature)
