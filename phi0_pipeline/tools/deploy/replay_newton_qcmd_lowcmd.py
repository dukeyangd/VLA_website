#!/usr/bin/env python3
"""Open-loop replay: Newton ``q_cmd`` (IL) → MJ remap → DDS LowCmd PD on MuJoCo.

Isolates MuJoCo physics/PD from ATM/student. No student, no ONNX decode.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import tyro
from dataclasses import dataclass

from phi0.deploy.lowcmd_publisher import LowCmdPublisher, LowStateReader
from phi0.deploy.sonic_decoder_obs import ISAACLAB_TO_MUJOCO, KDS_MJ, KPS_MJ


def _il_to_mj(q_il: np.ndarray) -> np.ndarray:
    q_il = np.asarray(q_il, dtype=np.float32)
    out = np.empty_like(q_il)
    if q_il.ndim == 1:
        for mj in range(29):
            out[mj] = q_il[int(ISAACLAB_TO_MUJOCO[mj])]
        return out
    for mj in range(29):
        out[..., mj] = q_il[..., int(ISAACLAB_TO_MUJOCO[mj])]
    return out


@dataclass
class ReplayConfig:
    newton_npz: Path
    rsi_npz: Path | None = None
    fps: float = 50.0
    start_delay_s: float = 0.3
    rsi_hold_frames: int = 10
    domain_id: int = 0
    dds_interface: str = "eth0"
    wait_g1_debug_s: float = 120.0
    out_npz: Path | None = None
    kp_scale: float = 1.0
    kd_scale: float = 1.0


def main(config: ReplayConfig) -> None:
    raw = np.load(config.newton_npz)
    q_il = raw["q_cmd"]
    if q_il.ndim == 3:
        q_il = q_il[:, 0]
    q_mj = _il_to_mj(q_il.astype(np.float32))
    T = int(q_mj.shape[0])

    iface = config.dds_interface.strip() or None
    lowstate = LowStateReader(domain_id=config.domain_id, interface=iface)
    lowcmd = LowCmdPublisher(
        domain_id=config.domain_id,
        interface=iface,
        init_factory=False,
        publish_hands=False,
        kp=(KPS_MJ * float(config.kp_scale)).astype(np.float32),
        kd=(KDS_MJ * float(config.kd_scale)).astype(np.float32),
    )
    print(
        f"[newton_replay] T={T} fps={config.fps} kp×{config.kp_scale:g} kd×{config.kd_scale:g} "
        f"npz={config.newton_npz}",
        flush=True,
    )
    lowstate.wait_ready(timeout_s=float(config.wait_g1_debug_s))
    snap0 = lowstate.snapshot()
    assert snap0 is not None
    lowcmd.set_mode_machine(int(snap0["mode_machine"]))

    # Trigger KEEP_POSE RSI restore, then hold RSI (or first Newton frame).
    lowcmd.publish(np.asarray(snap0["q_mj"], dtype=np.float32))
    time.sleep(max(0.25, float(config.start_delay_s)))
    if config.rsi_npz is not None and config.rsi_npz.is_file():
        pack = np.load(config.rsi_npz)
        # unified RSI export key
        key = "observation_qpos" if "observation_qpos" in pack.files else list(pack.files)[0]
        rsi_q = np.asarray(pack[key], dtype=np.float32).reshape(29)
        print(f"[newton_replay] RSI seed {config.rsi_npz} q0={float(rsi_q[0]):+.3f}", flush=True)
    else:
        rsi_q = q_mj[0].copy()
        print(f"[newton_replay] RSI seed from Newton q_cmd[0] q0={float(rsi_q[0]):+.3f}", flush=True)

    period = 1.0 / float(config.fps)
    for _ in range(max(0, int(config.rsi_hold_frames))):
        lowcmd.publish(rsi_q)
        time.sleep(period)

    q_meas_log: list[np.ndarray] = []
    t0 = time.monotonic()
    for i in range(T):
        target = t0 + (i + 1) * period
        snap = lowstate.snapshot()
        if snap is not None:
            lowcmd.set_mode_machine(int(snap["mode_machine"]))
            q_meas_log.append(np.asarray(snap["q_mj"], dtype=np.float32).copy())
        else:
            q_meas_log.append(np.full(29, np.nan, np.float32))
        lowcmd.publish(q_mj[i])
        if i == 0 or (i + 1) % 50 == 0:
            meas = q_meas_log[-1]
            track = float(np.nanmean((q_mj[i] - meas) ** 2))
            print(
                f"[newton_replay] t={i} track_mse={track:.5f} "
                f"q0_cmd={float(q_mj[i, 0]):+.3f} q0_meas={float(meas[0]):+.3f}",
                flush=True,
            )
        dt = target - time.monotonic()
        if dt > 0:
            time.sleep(dt)

    if config.out_npz is not None:
        config.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.out_npz,
            q_cmd=q_mj,
            q_meas=np.stack(q_meas_log, axis=0),
            q_cmd_il=q_il.astype(np.float32),
            source=str(config.newton_npz),
        )
        print(f"[newton_replay] wrote {config.out_npz}", flush=True)
    print(f"[newton_replay] done frames={T}", flush=True)


if __name__ == "__main__":
    # Allow KP scale via env without rewriting CLI.
    cfg = tyro.cli(ReplayConfig)
    if "PHI0_ATM_KP_SCALE" in os.environ:
        cfg.kp_scale = float(os.environ["PHI0_ATM_KP_SCALE"])
    if "PHI0_ATM_KD_SCALE" in os.environ:
        cfg.kd_scale = float(os.environ["PHI0_ATM_KD_SCALE"])
    main(cfg)
