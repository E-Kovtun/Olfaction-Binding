"""A live status block for long GPU sweeps, and the placement that fills the cards.

WHY. A sweep with a dozen workers printing three lines per cell produces a wall of
interleaved log that says nothing about the two questions actually being asked -- how
far along is this, and are the GPUs being used. This module answers both in a fixed
block that rewrites itself in place, and pushes the per-cell chatter into per-worker
log files where it stays available for a post-mortem without scrolling past anything.

WHAT IT IS NOT. It does not measure how much memory a training needs; that number
depends on the dataset, the graph and the batch, and the honest way to find it is to
raise `--per-gpu` and watch the memory row. The block exists to make that experiment
readable, not to replace it.

Nothing here imports torch: the parent process must NOT initialise CUDA, because the
children inherit `CUDA_VISIBLE_DEVICES` and a parent that has already opened a context
would pin itself to a card. GPU numbers come from `nvidia-smi`, which also reports
utilisation -- something a torch-side query cannot see.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time

_QUERY = "index,memory.used,memory.total,utilization.gpu"


def gpu_stats(indices=None):
    """{index: (used_mb, total_mb, util_pct)}. Empty when there is no nvidia-smi,
    which is the normal state on a CPU box and must not be an error."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    stats = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            i, used, total, util = (int(parts[0]), float(parts[1]), float(parts[2]),
                                    float(parts[3]))
        except ValueError:
            continue
        if indices is None or i in indices:
            stats[i] = (used, total, util)
    return stats


def visible_gpus():
    """Every GPU nvidia-smi can see, in index order. `[]` on a CPU box."""
    return sorted(gpu_stats())


def plan_placement(gpus, n_workers, free_mb=None):
    """Which GPU each worker is pinned to, in proportion to the memory actually FREE.

    Plain round-robin gives every card the same number of workers, which is wrong the
    moment one of them is already half-occupied -- by another user, or by a run of our
    own left behind. Dealing in proportion to free memory (largest-remainder, so the
    counts sum exactly) puts fewer workers on a busy card without ever having to be
    told about it. With equally free cards it degenerates to round-robin, which is the
    behaviour to expect on an idle box.

    Returns a list of length `n_workers`; entry i is the GPU index for worker i.
    """
    if not gpus:
        return []
    if free_mb is None:
        stats = gpu_stats(set(gpus))
        free_mb = [max(stats.get(g, (0.0, 1.0, 0.0))[1] - stats.get(g, (0.0, 1.0, 0.0))[0],
                       0.0) or 1.0 for g in gpus]
    total = sum(free_mb)
    if total <= 0:
        free_mb, total = [1.0] * len(gpus), float(len(gpus))
    raw = [n_workers * f / total for f in free_mb]
    cnt = [int(x) for x in raw]
    short = n_workers - sum(cnt)
    for i in sorted(range(len(gpus)), key=lambda i: raw[i] - cnt[i], reverse=True)[:short]:
        cnt[i] += 1
    # interleave rather than block, so the first workers to start -- the ones that run
    # while the queue is still full -- are spread over every card from the outset
    order, out = list(cnt), []
    while len(out) < n_workers:
        for i, g in enumerate(gpus):
            if order[i] > 0:
                out.append(g)
                order[i] -= 1
    return out[:n_workers]


def _bar(frac, width=28):
    filled = int(round(max(0.0, min(1.0, frac)) * width))
    return "#" * filled + "." * (width - filled)


def fmt_time(sec):
    if sec is None or sec != sec or sec < 0:
        return "--"
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


class Dashboard:
    """A fixed block that rewrites itself in place.

    Falls back to one progress line every `quiet_every` seconds when stdout is not a
    terminal -- which is what happens under `nohup`, in a pipe, or in CI, and where
    cursor movement would otherwise fill the file with escape codes."""

    def __init__(self, total, gpus, title="sweep", stream=None, refresh=2.0,
                 quiet_every=60.0, tty=None):
        self.total, self.gpus, self.title = max(int(total), 0), list(gpus), title
        self.stream = stream or sys.stdout
        self.refresh, self.quiet_every = refresh, quiet_every
        self.tty = self.stream.isatty() if tty is None else tty
        self.t0 = time.time()
        self.done = 0
        self.running = {}                 # worker id -> (label, started_at)
        self.worker_gpu = {}              # worker id -> gpu index
        self.last = ""                    # the most recent finished cell
        self.notes = []                   # lines that must survive a redraw
        self.stage = ""
        self._lines = 0
        self._last_render = 0.0
        self._gpu_cache, self._gpu_at = {}, 0.0

    # ---------------------------------------------------------------- events
    def bind(self, wid, gpu):
        self.worker_gpu[wid] = gpu

    def start_job(self, wid, label):
        self.running[wid] = (label, time.time())
        self.render()

    def finish_job(self, wid, label, summary="", counts=True):
        self.running.pop(wid, None)
        if counts:
            self.done += 1
        self.last = f"{label}  {summary}".strip()
        self.render()

    def note(self, text):
        """A line that must not be overwritten -- a failure, a file completed."""
        self.notes.append(text)
        self.notes[:] = self.notes[-4:]
        self.render(force=True)

    def set_stage(self, text):
        self.stage = text
        self.render(force=True)

    # ---------------------------------------------------------------- render
    def _gpus_now(self):
        if time.time() - self._gpu_at > max(self.refresh, 2.0):
            self._gpu_cache, self._gpu_at = gpu_stats(set(self.gpus)), time.time()
        return self._gpu_cache

    def _body(self):
        el = time.time() - self.t0
        rate = self.done / el if el > 0 and self.done else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else None
        pct = self.done / self.total if self.total else 0.0
        lines = [f"  {self.title}",
                 f"  [{_bar(pct)}] {self.done}/{self.total}  {pct * 100:5.1f}%"
                 f"   elapsed {fmt_time(el)}   ETA {fmt_time(eta)}"
                 f"   {rate * 60:.2f} cells/min"]
        if self.stage:
            lines.append(f"  now: {self.stage}")
        if self.last:
            lines.append(f"  last: {self.last[:110]}")
        stats = self._gpus_now()
        if self.gpus:
            lines.append("")
            lines.append(f"  {'gpu':<5}{'memory':<20}{'util':<7}{'w':<4}running")
            for g in self.gpus:
                used, total, util = stats.get(g, (0.0, 0.0, 0.0))
                mem = (f"{used / 1024:.1f}/{total / 1024:.1f} GB" if total
                       else "n/a")
                mine = [w for w, gg in self.worker_gpu.items() if gg == g]
                jobs = [self.running[w][0] for w in mine if w in self.running]
                lines.append(f"  {g:<5}{mem:<20}{util:>3.0f}%   {len(mine):<4}"
                             + " ".join(jobs)[:60])
        for n in self.notes:
            lines.append(f"  ! {n[:110]}")
        return lines

    def render(self, force=False):
        now = time.time()
        if not self.tty:
            if force or now - self._last_render >= self.quiet_every:
                self._last_render = now
                el = now - self.t0
                rate = self.done / el if el > 0 and self.done else 0.0
                eta = (self.total - self.done) / rate if rate > 0 else None
                print(f"  [{self.done}/{self.total}] {fmt_time(el)} elapsed, "
                      f"ETA {fmt_time(eta)} -- {self.last[:80]}",
                      file=self.stream, flush=True)
            return
        if not force and now - self._last_render < self.refresh:
            return
        self._last_render = now
        body = self._body()
        out = []
        if self._lines:
            out.append(f"\033[{self._lines}A")
        for line in body:
            out.append("\033[2K" + line + "\n")
        self._lines = len(body)
        self.stream.write("".join(out))
        self.stream.flush()

    def close(self):
        self.render(force=True)
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()
        self._lines = 0
