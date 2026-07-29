#!/usr/bin/env python3
"""
compare_pruna_vae.py — stock vs PrunaVAED decoder, ~4 s clip
============================================================
Spawns ``server.py`` twice (stock, then pruna), runs the same
``videofentanyl.py`` job each time, and prints a side-by-side timing
summary plus ``COMPARE_JSON:{...}``.

Default workload: **97 frames @ 24 fps ≈ 4.0 s**, 8 distilled steps, fixed seed.

Requires branch ``pruna-vae-decoder`` (or equivalent) with ``--vae-decoder``
support, and a venv with ltx-2-mlx (see README). Pruna weights download from
``audiohacking/pruna-vaed-mlx`` on first ``--vae-decoder pruna`` load.

Examples
--------
  # Full A/B on this machine (free port 8765)
  ./scripts/compare_pruna_vae.py

  # Custom prompt / seed / port
  ./scripts/compare_pruna_vae.py -p "neon alley rain" --seed 42 --port 9000

  # Verbose server logs (useful when Pruna Hub download runs)
  ./scripts/compare_pruna_vae.py --verbose-server

  # Dry-run (print commands only)
  ./scripts/compare_pruna_vae.py --dry-run

Manual two-terminal recipe (same comparison without this script)
----------------------------------------------------------------
  # Terminal A — stock
  python server.py --port 8765 --num-frames 97 --infer-steps 8 --vae-decoder stock
  python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
      --prompt "cinematic drone shot over misty mountains at sunrise" \\
      --seed 42 --num-frames 97 --output-dir ./compare_pruna_runs --prefix stock

  # Stop server (Ctrl+C), then Terminal A — pruna
  python server.py --port 8765 --num-frames 97 --infer-steps 8 --vae-decoder pruna
  python videofentanyl.py --server ws://127.0.0.1:8765/ws \\
      --prompt "cinematic drone shot over misty mountains at sunrise" \\
      --seed 42 --num-frames 97 --output-dir ./compare_pruna_runs --prefix pruna

Report back: the printed summary table, both MP4 paths, and the COMPARE_JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_PY = REPO_ROOT / "server.py"
CLIENT_PY = REPO_ROOT / "videofentanyl.py"

# ~4.04 s @ 24 fps; LTX requires 8k+1 frames.
DEFAULT_NUM_FRAMES = 97
DEFAULT_INFER_STEPS = 8
DEFAULT_FPS = 24
DEFAULT_PORT = 8765
DEFAULT_SEED = 42
DEFAULT_PROMPT = "cinematic drone shot over misty mountains at sunrise, slow forward motion"

LATENCY_RE = re.compile(
    r"← latency\s+gen=([0-9.]+)ms\s+e2e=([0-9.]+)ms",
)
SAVED_RE = re.compile(
    r"✓ saved\s+(\S+)\s+\((\d+) KB.*,\s*([0-9.]+)s\)",
)


def resolve_interpreter(
    repo: Path,
    explicit: Path | None,
    allow_system: bool,
) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            print(f"Error: --python not found: {p}", file=sys.stderr)
            sys.exit(2)
        return p

    bindir = repo / ".venv" / "bin"
    for name in ("python3", "python"):
        cand = bindir / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()

    if allow_system:
        return Path(sys.executable).resolve()

    print(
        "Error: no virtualenv at ./.venv/bin/python3\n"
        "See README « Local server (Apple Silicon / MLX) », then re-run.\n"
        "Escape hatch: --allow-system-python\n",
        file=sys.stderr,
    )
    sys.exit(2)


def wait_tcp(host: str, port: int, timeout_s: float, interval_s: float = 0.25) -> float:
    deadline = time.monotonic() + timeout_s
    t0 = time.monotonic()
    last_err: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return time.monotonic() - t0
        except OSError as e:
            last_err = e
            time.sleep(interval_s)
    msg = f"timeout waiting for {host}:{port} ({timeout_s:.0f}s)"
    if last_err is not None:
        msg += f"  last error: {last_err}"
    raise TimeoutError(msg)


def stop_server_gracefully(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def parse_client_output(stdout: str) -> dict:
    out: dict = {
        "generation_ms": None,
        "e2e_ms": None,
        "job_elapsed_s": None,
        "saved_path": None,
        "saved_kb": None,
    }
    for line in stdout.splitlines():
        m = LATENCY_RE.search(line)
        if m:
            out["generation_ms"] = float(m.group(1))
            out["e2e_ms"] = float(m.group(2))
        m = SAVED_RE.search(line)
        if m:
            out["saved_path"] = m.group(1)
            out["saved_kb"] = int(m.group(2))
            out["job_elapsed_s"] = float(m.group(3))
    return out


def newest_mp4(directory: Path, prefix: str) -> Path | None:
    matches = sorted(
        directory.glob(f"{prefix}*.mp4"),
        key=lambda p: p.stat().st_mtime if p.is_file() else 0,
        reverse=True,
    )
    return matches[0] if matches else None


def run_leg(
    *,
    py: Path,
    port: int,
    vae_decoder: str,
    prompt: str,
    seed: int,
    num_frames: int,
    infer_steps: int,
    fps: int,
    model: str | None,
    height: int | None,
    width: int | None,
    out_dir: Path,
    prefix: str,
    ready_timeout: float,
    verbose_server: bool,
    dry_run: bool,
) -> dict:
    server_url = f"ws://127.0.0.1:{port}/ws"
    server_cmd = [
        str(py),
        str(SERVER_PY),
        "--port", str(port),
        "--num-frames", str(num_frames),
        "--infer-steps", str(infer_steps),
        "--fps", str(fps),
        "--vae-decoder", vae_decoder,
        "--no-web-ui",
    ]
    if model:
        server_cmd.extend(["--model", model])

    client_cmd = [
        str(py),
        str(CLIENT_PY),
        "--mode", "ltx",
        "--server", server_url,
        "--prompt", prompt,
        "--seed", str(seed),
        "--num-frames", str(num_frames),
        "--output-dir", str(out_dir),
        "--prefix", prefix,
        "--delay", "0",
    ]
    if height is not None:
        client_cmd.extend(["--height", str(height)])
    if width is not None:
        client_cmd.extend(["--width", str(width)])

    print(f"\n{'═' * 60}", flush=True)
    print(f"  LEG: vae-decoder={vae_decoder}  prefix={prefix}", flush=True)
    print(f"{'═' * 60}", flush=True)

    if dry_run:
        print("[dry-run] server:", subprocess.list2cmdline(server_cmd))
        print("[dry-run] client:", subprocess.list2cmdline(client_cmd))
        return {
            "vae_decoder": vae_decoder,
            "dry_run": True,
            "server_cmd": server_cmd,
            "client_cmd": client_cmd,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_dest = None if verbose_server else subprocess.PIPE
    stderr_dest = None if verbose_server else subprocess.STDOUT

    t_server0 = time.perf_counter()
    server_proc = subprocess.Popen(
        server_cmd,
        cwd=str(REPO_ROOT),
        stdout=stdout_dest,
        stderr=stderr_dest,
        text=True,
    )
    server_log = ""
    try:
        ready_s = wait_tcp("127.0.0.1", port, timeout_s=ready_timeout)
    except TimeoutError as e:
        if server_proc.stdout is not None:
            try:
                server_log = server_proc.stdout.read() or ""
            except Exception:
                pass
        print(f"Error: {e}", file=sys.stderr)
        if server_proc.poll() is not None:
            print(f"  server exited early (code {server_proc.returncode})", file=sys.stderr)
        if server_log:
            sys.stderr.write(server_log[-4000:])
        stop_server_gracefully(server_proc)
        return {
            "vae_decoder": vae_decoder,
            "ok": False,
            "error": str(e),
            "server_ready_tcp_s": None,
            "client_exit_code": None,
        }

    server_ready_wall_s = time.perf_counter() - t_server0
    print(
        f"  server ready  tcp_wait={ready_s:.1f}s  wall={server_ready_wall_s:.1f}s",
        flush=True,
    )

    t_client0 = time.perf_counter()
    try:
        cp = subprocess.run(
            client_cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=ready_timeout + 7200.0,
        )
    except subprocess.TimeoutExpired:
        print("Error: videofentanyl timed out", file=sys.stderr)
        stop_server_gracefully(server_proc)
        return {
            "vae_decoder": vae_decoder,
            "ok": False,
            "error": "client_timeout",
            "server_ready_tcp_s": ready_s,
            "server_ready_wall_s": server_ready_wall_s,
        }
    client_wall_s = time.perf_counter() - t_client0

    if cp.stdout:
        sys.stdout.write(cp.stdout)
        if not cp.stdout.endswith("\n"):
            print()
    if cp.stderr:
        sys.stderr.write(cp.stderr)

    parsed = parse_client_output(cp.stdout or "")
    mp4 = None
    if parsed["saved_path"]:
        cand = Path(parsed["saved_path"])
        mp4 = cand if cand.is_file() else out_dir / cand.name
    if mp4 is None or not mp4.is_file():
        mp4 = newest_mp4(out_dir, prefix)

    bytes_out = mp4.stat().st_size if mp4 and mp4.is_file() else None

    stop_server_gracefully(server_proc)

    return {
        "vae_decoder": vae_decoder,
        "ok": cp.returncode == 0 and mp4 is not None and mp4.is_file(),
        "server_ready_tcp_s": ready_s,
        "server_ready_wall_s": server_ready_wall_s,
        "client_subprocess_wall_s": client_wall_s,
        "generation_ms": parsed["generation_ms"],
        "e2e_ms": parsed["e2e_ms"],
        "job_elapsed_s": parsed["job_elapsed_s"],
        "client_exit_code": cp.returncode,
        "output_path": str(mp4) if mp4 else None,
        "output_bytes": bytes_out,
        "prefix": prefix,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compare ~4s local generations with stock vs PrunaVAED VAE decoder."
        ),
    )
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument(
        "--model",
        default=None,
        help="Forwarded to server.py --model (default: server auto/default)",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"default {DEFAULT_NUM_FRAMES} (~4s @ 24fps)",
    )
    p.add_argument("--infer-steps", type=int, default=DEFAULT_INFER_STEPS)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("-p", "--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "compare_pruna_runs",
    )
    p.add_argument(
        "--order",
        choices=["stock-first", "pruna-first"],
        default="stock-first",
        help="Which decoder leg to run first (default: stock-first)",
    )
    p.add_argument(
        "--ready-timeout",
        type=float,
        default=1200.0,
        help="Max wait for server TCP (includes Pruna Hub download on first run)",
    )
    p.add_argument("--verbose-server", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--python", type=Path, default=None)
    p.add_argument("--allow-system-python", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    py = resolve_interpreter(
        REPO_ROOT, args.python, allow_system=args.allow_system_python
    )
    if not SERVER_PY.is_file() or not CLIENT_PY.is_file():
        print("Error: missing server.py or videofentanyl.py", file=sys.stderr)
        return 2

    out_dir = args.output_dir.expanduser().resolve()
    print(f"Using interpreter: {py}", flush=True)
    print(f"Output dir:        {out_dir}", flush=True)
    print(
        f"Workload:          frames={args.num_frames} steps={args.infer_steps} "
        f"seed={args.seed} fps={args.fps}",
        flush=True,
    )
    print(f"Prompt:            {args.prompt}", flush=True)

    legs = ("stock", "pruna")
    if args.order == "pruna-first":
        legs = ("pruna", "stock")

    results: list[dict] = []
    for vae in legs:
        results.append(
            run_leg(
                py=py,
                port=args.port,
                vae_decoder=vae,
                prompt=args.prompt,
                seed=args.seed,
                num_frames=args.num_frames,
                infer_steps=args.infer_steps,
                fps=args.fps,
                model=args.model,
                height=args.height,
                width=args.width,
                out_dir=out_dir,
                prefix=f"compare_{vae}",
                ready_timeout=args.ready_timeout,
                verbose_server=args.verbose_server,
                dry_run=args.dry_run,
            )
        )

    if args.dry_run:
        return 0

    by_name = {r["vae_decoder"]: r for r in results}
    stock = by_name.get("stock", {})
    pruna = by_name.get("pruna", {})

    def _fmt_ms(v: object) -> str:
        return f"{float(v):.0f} ms" if v is not None else "—"

    def _fmt_s(v: object) -> str:
        return f"{float(v):.2f}s" if v is not None else "—"

    print()
    print("  ── stock vs pruna summary ──")
    print(f"  {'metric':<28} {'stock':>14} {'pruna':>14}")
    print(f"  {'-'*28} {'-'*14} {'-'*14}")
    print(
        f"  {'server ready (wall)':<28} "
        f"{_fmt_s(stock.get('server_ready_wall_s')):>14} "
        f"{_fmt_s(pruna.get('server_ready_wall_s')):>14}"
    )
    print(
        f"  {'client wall':<28} "
        f"{_fmt_s(stock.get('client_subprocess_wall_s')):>14} "
        f"{_fmt_s(pruna.get('client_subprocess_wall_s')):>14}"
    )
    print(
        f"  {'generation_ms':<28} "
        f"{_fmt_ms(stock.get('generation_ms')):>14} "
        f"{_fmt_ms(pruna.get('generation_ms')):>14}"
    )
    print(
        f"  {'e2e_ms':<28} "
        f"{_fmt_ms(stock.get('e2e_ms')):>14} "
        f"{_fmt_ms(pruna.get('e2e_ms')):>14}"
    )
    print(
        f"  {'job_elapsed_s':<28} "
        f"{_fmt_s(stock.get('job_elapsed_s')):>14} "
        f"{_fmt_s(pruna.get('job_elapsed_s')):>14}"
    )
    for name, leg in (("stock", stock), ("pruna", pruna)):
        print(f"  {name} mp4: {leg.get('output_path') or '(missing)'}")
    print()

    # Speedup on client wall / e2e when both present
    speedups: dict[str, float | None] = {"client_wall": None, "e2e_ms": None}
    if stock.get("client_subprocess_wall_s") and pruna.get("client_subprocess_wall_s"):
        speedups["client_wall"] = (
            float(stock["client_subprocess_wall_s"])
            / float(pruna["client_subprocess_wall_s"])
        )
        print(
            f"  client_wall speedup (stock/pruna): {speedups['client_wall']:.3f}×",
        )
    if stock.get("e2e_ms") and pruna.get("e2e_ms"):
        speedups["e2e_ms"] = float(stock["e2e_ms"]) / float(pruna["e2e_ms"])
        print(f"  e2e_ms speedup (stock/pruna):      {speedups['e2e_ms']:.3f}×")
    print()

    record = {
        "version": 1,
        "ts": datetime.now(timezone.utc).isoformat(),
        "config": {
            "python": str(py),
            "num_frames": args.num_frames,
            "infer_steps": args.infer_steps,
            "fps": args.fps,
            "seed": args.seed,
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "model": args.model,
            "order": args.order,
            "approx_duration_s": round(args.num_frames / float(args.fps), 3),
        },
        "legs": results,
        "speedups": speedups,
    }
    print("COMPARE_JSON:" + json.dumps(record, separators=(",", ":")))

    ok = all(r.get("ok") for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
