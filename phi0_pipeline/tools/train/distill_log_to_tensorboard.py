#!/usr/bin/env python3
import argparse
import re
import time
from pathlib import Path

from tensorboardX import SummaryWriter

PAT_MAIN = re.compile(r"\[distill\]\s+(\d+)/(\d+).*?loss=([0-9.]+)\s+Lz=([0-9.]+)\s+Lq=([0-9.]+)\s+Ls=([0-9.]+)")
PAT_SPS = re.compile(r"([0-9.]+)\s+step/s")
PAT_ETA = re.compile(r"ETA\s+([0-9.]+)h")


def latest_exp_dir(root: Path) -> Path:
    cands = sorted(root.glob("sonic_online_distill_protomotions_full8gpu_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No experiment folder found under {root}")
    return cands[0]


def emit_line(writer: SummaryWriter, line: str, last_step: int) -> int:
    m = PAT_MAIN.search(line)
    if not m:
        return last_step
    step = int(m.group(1))
    total = int(m.group(2))
    loss = float(m.group(3))
    lz = float(m.group(4))
    lq = float(m.group(5))
    ls = float(m.group(6))
    sps_m = PAT_SPS.search(line)
    eta_m = PAT_ETA.search(line)
    sps = float(sps_m.group(1)) if sps_m else None
    eta_h = float(eta_m.group(1)) if eta_m else None
    if step <= last_step:
        return last_step

    writer.add_scalar("distill/loss", loss, step)
    writer.add_scalar("distill/Lz", lz, step)
    writer.add_scalar("distill/Lq", lq, step)
    writer.add_scalar("distill/Ls", ls, step)
    if sps is not None:
        writer.add_scalar("distill/step_per_sec", sps, step)
    if eta_h is not None:
        writer.add_scalar("distill/eta_hours", eta_h, step)
    writer.add_scalar("distill/progress", step / max(total, 1), step)
    return step


def backfill(writer: SummaryWriter, log_path: Path) -> int:
    last_step = -1
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            last_step = emit_line(writer, line, last_step)
    writer.flush()
    return last_step


def follow(writer: SummaryWriter, log_path: Path, last_step: int, poll_s: float) -> None:
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_s)
                continue
            last_step = emit_line(writer, line, last_step)
            if last_step >= 0:
                writer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert distill run.log to TensorBoard scalars.")
    parser.add_argument("--experiments-root", default="/mnt/data2/wpy/workspace/phi-0-wbc/experiments")
    parser.add_argument("--exp-dir", default="", help="Experiment directory. Empty means latest.")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tb-dir", default="", help="Output TensorBoard log dir. Empty means <exp>/tensorboard/rank{rank}.")
    parser.add_argument("--follow", action="store_true", help="Keep following appended log lines.")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(args.experiments_root)
    exp_dir = Path(args.exp_dir) if args.exp_dir else latest_exp_dir(root)
    log_path = exp_dir / f"rank{args.rank}" / "run.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    tb_dir = Path(args.tb_dir) if args.tb_dir else exp_dir / "tensorboard" / f"rank{args.rank}"
    tb_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))
    last_step = backfill(writer, log_path)
    print(f"exp_dir={exp_dir}")
    print(f"log={log_path}")
    print(f"tb_dir={tb_dir}")
    print(f"last_step={last_step}")

    if args.follow:
        follow(writer, log_path, last_step, args.poll_seconds)


if __name__ == "__main__":
    main()
