#!/usr/bin/env python3
"""
Central GPU inference server for self-play.

A single server process owns the model on GPU. Worker processes encode boards
to numpy tensors locally (CPU work, parallel across workers) and submit them
through an mp.Queue. The server coalesces incoming requests into one
super-batch per forward pass and returns (values, policies) to each worker
through a per-worker reply queue.

Why: with 24 self-play workers each owning their own copy of the model,
we observed CPU at ~98% and GPU at ~95% but throughput stuck at ~31 mps —
GPU dispatch contention from 24 separate CUDA contexts is the wall. A
single-context server amortizes kernel launches and runs larger fused
batches, which is what the H100 actually wants.

This file is standalone. It does not modify any existing code. It exposes:

    InferenceServer:
        start() / stop() / make_client(worker_id) / stats()

    InferenceClient (returned by make_client):
        forward_np(x_np) -> (values_np, policies_np)
        x_np: (B, 18, 8, 8) float32 numpy array
        values_np: (B,) float32, policies_np: (B, 4672) float32

Run a standalone benchmark:

    ~/myvenv/bin/python inference_server.py --model <path> \
        --workers 24 --duration 30 --avg-batch 16 --max-batch 1024

The benchmark spawns N fake worker processes that pump random tensors
through the server and reports leaves/sec, super-batches/sec, mean batch.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from contextlib import nullcontext
from queue import Empty
from typing import Optional, Tuple

import numpy as np
import torch

from chess_engine import ChessNet


# Bucket sizes used to pad the super-batch shape, so cudnn.benchmark sees
# only a small number of distinct shapes (one cached kernel each).
_PAD_BUCKETS = (1, 8, 16, 32, 64, 128, 256, 512, 1024)


def _next_bucket(n: int, max_batch: int) -> int:
    for b in _PAD_BUCKETS:
        if b >= n and b <= max_batch:
            return b
    return max_batch


def _server_loop(
    model_path: str,
    device_str: str,
    num_workers: int,
    max_batch: int,
    coalesce_us: int,
    pad_to_buckets: bool,
    request_q,
    reply_qs,
    ready_evt,
    shutdown_evt,
    stats_q,
):
    """Server process main loop.

    Protocol:
      request:  (worker_id: int, req_id: int, x_np: float32 ndarray (B,18,8,8))
      reply:    (req_id: int, values_np: float32 (B,), policies_np: float32 (B,P))
    """
    # Loaded inside the child to avoid CUDA fork issues.
    import torch  # noqa: F401  (re-import within child for clarity)

    device = torch.device(device_str)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=16)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        model = model.to(dtype=torch.bfloat16)
    model.eval()

    # Pinned host scratch for fast HtoD; sized to max_batch.
    if device.type == "cuda":
        pinned = torch.empty(
            (max_batch, 18, 8, 8), dtype=torch.float32, pin_memory=True
        )
    else:
        pinned = None

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    # Warmup at each bucket size to populate cudnn kernel cache and compile
    # any lazy paths before workers start hammering us.
    with torch.inference_mode(), amp_ctx:
        warm_buckets = [b for b in _PAD_BUCKETS if b <= max_batch]
        for wb in warm_buckets:
            x = torch.zeros((wb, 18, 8, 8), dtype=torch.float32, device=device)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    ready_evt.set()

    # Stats
    n_super = 0
    n_reqs = 0
    n_leaves = 0
    n_padded_leaves = 0
    t_forward = 0.0
    t_xfer = 0.0
    t_coalesce = 0.0
    t_split = 0.0
    t_first_get = 0.0
    t_start = time.time()

    coalesce_s = max(0, coalesce_us) * 1e-6

    while not shutdown_evt.is_set():
        # Block on first request (with periodic shutdown poll).
        t_g0 = time.perf_counter()
        try:
            first = request_q.get(timeout=0.1)
        except Empty:
            continue
        t_first_get += time.perf_counter() - t_g0

        items = [first]
        total = first[2].shape[0]

        # Coalesce: drain whatever is queued; optionally wait coalesce_us
        # for stragglers. wait=0 gives lowest latency; wait>0 helps only if
        # producers are bursty.
        t_c0 = time.perf_counter()
        if coalesce_s > 0.0:
            deadline = t_c0 + coalesce_s
            while total < max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    it = request_q.get(timeout=min(remaining, 0.0005))
                except Empty:
                    continue
                items.append(it)
                total += it[2].shape[0]
        else:
            while total < max_batch:
                try:
                    it = request_q.get_nowait()
                except Empty:
                    break
                items.append(it)
                total += it[2].shape[0]
        t_coalesce += time.perf_counter() - t_c0

        # Build super-batch.
        if len(items) == 1:
            x_np = items[0][2]
        else:
            x_np = np.concatenate([it[2] for it in items], axis=0)

        actual = x_np.shape[0]
        if actual > max_batch:
            # Defensive: shouldn't happen since we cap at max_batch above.
            x_np = x_np[:max_batch]
            actual = max_batch

        # Forward.
        with torch.inference_mode(), amp_ctx:
            t_x0 = time.perf_counter()
            if pinned is not None:
                if pad_to_buckets:
                    pad_to = _next_bucket(actual, max_batch)
                else:
                    pad_to = actual
                # Copy real rows into pinned host buffer.
                pinned[:actual].copy_(torch.from_numpy(x_np))
                if pad_to > actual:
                    # Pad by repeating the last row (matches existing logic).
                    pinned[actual:pad_to].copy_(
                        pinned[actual - 1 : actual].expand(pad_to - actual, -1, -1, -1)
                    )
                x_t = pinned[:pad_to].to(device, non_blocking=True)
                x_t = x_t.contiguous(memory_format=torch.channels_last)
            else:
                pad_to = actual
                x_t = torch.from_numpy(x_np).to(device)
            t_xfer += time.perf_counter() - t_x0

            t_f0 = time.perf_counter()
            values, policies = model(x_t)
            # Trim padding before serializing back.
            values = values[:actual]
            policies = policies[:actual]
            v_np = values.float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
            p_np = policies.float().cpu().numpy().astype(np.float32, copy=False)
            t_forward += time.perf_counter() - t_f0

        # Split + reply.
        t_s0 = time.perf_counter()
        offset = 0
        for it in items:
            wid = it[0]
            rid = it[1]
            sz = it[2].shape[0]
            v_chunk = v_np[offset : offset + sz]
            p_chunk = p_np[offset : offset + sz]
            reply_qs[wid].put((rid, v_chunk, p_chunk))
            offset += sz
        t_split += time.perf_counter() - t_s0

        n_super += 1
        n_reqs += len(items)
        n_leaves += int(actual)
        n_padded_leaves += int(pad_to)

    # Final stats dump.
    elapsed = time.time() - t_start
    if stats_q is not None:
        stats_q.put(
            dict(
                super_batches=n_super,
                requests=n_reqs,
                leaves=n_leaves,
                padded_leaves=n_padded_leaves,
                forward_time=t_forward,
                xfer_time=t_xfer,
                coalesce_time=t_coalesce,
                split_time=t_split,
                first_get_time=t_first_get,
                elapsed=elapsed,
            )
        )


class InferenceServer:
    def __init__(
        self,
        model_path: str,
        num_workers: int,
        device: str = "cuda",
        max_batch: int = 1024,
        coalesce_us: int = 0,
        pad_to_buckets: bool = True,
        ctx: Optional[mp.context.BaseContext] = None,
    ):
        if ctx is None:
            ctx = mp.get_context("spawn")
        self.ctx = ctx
        self.model_path = model_path
        self.num_workers = num_workers
        self.device = device
        self.max_batch = max_batch
        self.coalesce_us = coalesce_us
        self.pad_to_buckets = pad_to_buckets

        # Bound the request queue so a runaway producer can't OOM us.
        self.request_q = ctx.Queue(maxsize=num_workers * 4)
        self.reply_qs = [ctx.Queue() for _ in range(num_workers)]
        self.ready_evt = ctx.Event()
        self.shutdown_evt = ctx.Event()
        self.stats_q = ctx.Queue()
        self.proc: Optional[mp.process.BaseProcess] = None

    def start(self, ready_timeout: float = 180.0):
        self.proc = self.ctx.Process(
            target=_server_loop,
            args=(
                self.model_path,
                self.device,
                self.num_workers,
                self.max_batch,
                self.coalesce_us,
                self.pad_to_buckets,
                self.request_q,
                self.reply_qs,
                self.ready_evt,
                self.shutdown_evt,
                self.stats_q,
            ),
            daemon=True,
            name="InferenceServer",
        )
        self.proc.start()
        if not self.ready_evt.wait(timeout=ready_timeout):
            self.stop()
            raise RuntimeError(
                f"InferenceServer did not become ready within {ready_timeout}s"
            )

    def stop(self, timeout: float = 10.0):
        self.shutdown_evt.set()
        if self.proc is not None:
            self.proc.join(timeout=timeout)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=2.0)
            self.proc = None

    def make_client(self, worker_id: int) -> "InferenceClient":
        if not (0 <= worker_id < self.num_workers):
            raise ValueError(
                f"worker_id {worker_id} out of range [0,{self.num_workers})"
            )
        return InferenceClient(worker_id, self.request_q, self.reply_qs[worker_id])

    def stats(self) -> Optional[dict]:
        try:
            return self.stats_q.get(timeout=1.0)
        except Empty:
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class InferenceClient:
    """Lightweight handle held by a worker process.

    Single outstanding request per client (matches MCTS access pattern).
    Not thread-safe; use one client per worker process.
    """

    __slots__ = ("worker_id", "request_q", "reply_q", "_req_id")

    def __init__(self, worker_id: int, request_q, reply_q):
        self.worker_id = worker_id
        self.request_q = request_q
        self.reply_q = reply_q
        self._req_id = 0

    def forward_np(
        self, x_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if x_np.dtype != np.float32:
            x_np = np.ascontiguousarray(x_np, dtype=np.float32)
        elif not x_np.flags["C_CONTIGUOUS"]:
            x_np = np.ascontiguousarray(x_np)

        rid = self._req_id
        self._req_id = (self._req_id + 1) & 0x7FFFFFFF
        self.request_q.put((self.worker_id, rid, x_np))
        reply_rid, v_np, p_np = self.reply_q.get()
        if reply_rid != rid:
            raise RuntimeError(
                f"reply id mismatch (worker {self.worker_id}): "
                f"expected {rid}, got {reply_rid}"
            )
        return v_np, p_np


# =============================================================================
# Standalone benchmark
# =============================================================================
#
# Spawns N fake worker processes, each pumping random board tensors through
# the server. Measures end-to-end leaves/sec and per-worker latency.
# =============================================================================


def _bench_worker(
    worker_id: int,
    request_q,
    reply_q,
    avg_batch: int,
    duration: float,
    barrier,
    result_q,
):
    rng = np.random.default_rng(worker_id * 7919 + 1)
    client = InferenceClient(worker_id, request_q, reply_q)

    # Pre-allocate one buffer per likely batch size (avoids alloc churn in hot
    # loop). avg_batch is the mean; sample sizes 1..2*avg_batch-1 uniformly.
    barrier.wait()

    t_end = time.perf_counter() + duration
    n_calls = 0
    n_leaves = 0
    lat_sum = 0.0
    lat_max = 0.0

    while time.perf_counter() < t_end:
        b = max(1, int(rng.integers(1, max(2, 2 * avg_batch))))
        x = rng.standard_normal((b, 18, 8, 8), dtype=np.float32)

        t0 = time.perf_counter()
        v, p = client.forward_np(x)
        dt = time.perf_counter() - t0

        if v.shape[0] != b or p.shape[0] != b:
            raise RuntimeError(
                f"shape mismatch: req {b}, got v={v.shape}, p={p.shape}"
            )

        n_calls += 1
        n_leaves += b
        lat_sum += dt
        if dt > lat_max:
            lat_max = dt

    result_q.put(
        dict(
            worker_id=worker_id,
            calls=n_calls,
            leaves=n_leaves,
            lat_sum=lat_sum,
            lat_max=lat_max,
        )
    )


def _bench_main():
    parser = argparse.ArgumentParser(description="Benchmark the inference server.")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--avg-batch", type=int, default=16,
                        help="Mean per-request batch size (each worker)")
    parser.add_argument("--max-batch", type=int, default=1024,
                        help="Max super-batch size on the server")
    parser.add_argument("--coalesce-us", type=int, default=0,
                        help="Microseconds to wait for stragglers after first req")
    parser.add_argument("--no-bucket-pad", action="store_true",
                        help="Disable padding super-batches to bucket sizes")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[bench] model={args.model}")
    print(f"[bench] workers={args.workers} duration={args.duration}s "
          f"avg_batch={args.avg_batch} max_batch={args.max_batch} "
          f"coalesce_us={args.coalesce_us} bucket_pad={not args.no_bucket_pad}")

    ctx = mp.get_context("spawn")
    server = InferenceServer(
        model_path=args.model,
        num_workers=args.workers,
        device=args.device,
        max_batch=args.max_batch,
        coalesce_us=args.coalesce_us,
        pad_to_buckets=not args.no_bucket_pad,
        ctx=ctx,
    )

    print("[bench] starting server (loading model, warming up)...")
    t0 = time.time()
    server.start()
    print(f"[bench] server ready in {time.time()-t0:.1f}s")

    barrier = ctx.Barrier(args.workers + 1)
    result_q = ctx.Queue()
    procs = []
    for wid in range(args.workers):
        p = ctx.Process(
            target=_bench_worker,
            args=(
                wid,
                server.request_q,
                server.reply_qs[wid],
                args.avg_batch,
                args.duration,
                barrier,
                result_q,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    print(f"[bench] {args.workers} workers spawned; releasing barrier...")
    barrier.wait()
    bench_t0 = time.perf_counter()

    for p in procs:
        p.join()
    bench_elapsed = time.perf_counter() - bench_t0

    results = []
    while True:
        try:
            results.append(result_q.get_nowait())
        except Empty:
            break

    total_calls = sum(r["calls"] for r in results)
    total_leaves = sum(r["leaves"] for r in results)
    mean_lat_ms = 1000 * sum(r["lat_sum"] for r in results) / max(1, total_calls)
    max_lat_ms = 1000 * max((r["lat_max"] for r in results), default=0.0)

    print()
    print("=" * 64)
    print(f"  Wall: {bench_elapsed:.2f}s")
    print(f"  Total client calls   : {total_calls:,}  ({total_calls/bench_elapsed:.0f} /s)")
    print(f"  Total leaves served  : {total_leaves:,}  ({total_leaves/bench_elapsed:.0f} /s)")
    print(f"  Mean per-call latency: {mean_lat_ms:.2f} ms")
    print(f"  Max  per-call latency: {max_lat_ms:.2f} ms")
    print()

    server.stop()
    server_stats = server.stats()
    if server_stats is not None:
        ss = server_stats
        n_super = max(1, ss["super_batches"])
        print(f"  Server super-batches : {ss['super_batches']:,}  "
              f"({ss['super_batches']/bench_elapsed:.1f} /s)")
        print(f"  Server requests      : {ss['requests']:,}  "
              f"(avg {ss['requests']/n_super:.1f} reqs / super-batch)")
        print(f"  Mean real super-batch: {ss['leaves']/n_super:.1f}")
        print(f"  Mean padded batch    : {ss['padded_leaves']/n_super:.1f}")
        print(f"  Server forward time  : {ss['forward_time']:.2f}s "
              f"({100*ss['forward_time']/ss['elapsed']:.1f}% of server life)")
        print(f"  Server xfer time     : {ss['xfer_time']:.2f}s")
        print(f"  Server coalesce time : {ss['coalesce_time']:.2f}s")
        print(f"  Server split  time   : {ss['split_time']:.2f}s")
        print(f"  Server first-get time: {ss['first_get_time']:.2f}s "
              f"(waiting for any request)")
    print("=" * 64)


if __name__ == "__main__":
    _bench_main()
