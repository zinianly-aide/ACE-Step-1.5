#!/usr/bin/env python3
"""Locally assemble a timed 3:00 entrance cue from user-supplied audio + ACE-Step v3.

This recipe never downloads or embeds the AMIGO recording. Do not publish
the resulting mix without the necessary music redistribution permissions.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

SR = 44100
CHANNELS = 2
TOTAL = 180
INTRO = 10.0
STORM_START = 69.1
TARGET_LUFS = -16.0
TARGET_TP = -1.0


def run(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stderr + result.stdout


def duration(path: Path) -> float:
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], text=True).strip())


def decode(path: Path) -> np.ndarray:
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-vn",
        "-ar", str(SR), "-ac", str(CHANNELS),
        "-f", "f32le", "pipe:1"
    ])
    return np.frombuffer(raw, dtype="<f4").reshape(-1, CHANNELS).copy()


def save_wav(path: Path, audio: np.ndarray) -> None:
    proc = subprocess.Popen([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "f32le", "-ar", str(SR), "-ac", str(CHANNELS),
        "-i", "pipe:0", "-c:a", "pcm_s16le", str(path)
    ], stdin=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        # Chunk writes avoid excessive buffering and giant pipe arguments.
        for offset in range(0, len(audio), SR * 5):
            proc.stdin.write(audio[offset:offset + SR * 5].astype("<f4").tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while writing {path}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def loudness(path: Path) -> dict:
    proc = subprocess.run([
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
        "-af", "loudnorm=I=-16:TP=-1:LRA=11:print_format=json",
        "-f", "null", "-"
    ], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-3000:])
    match = re.findall(r'\{\s*"input_i"\s*:.*?\}', proc.stderr, flags=re.S)
    if not match:
        raise RuntimeError("Cannot parse loudnorm diagnostics")
    stats = json.loads(match[-1])
    return {
        "lufs": float(stats["input_i"]),
        "true_peak_dbtp": float(stats["input_tp"]),
        "lra": float(stats["input_lra"])
    }


def mix(amigo: np.ndarray, v3: np.ndarray) -> np.ndarray:
    n = TOTAL * SR
    if len(v3) != n:
        raise ValueError(f"v3 must be precisely 180s, got {len(v3) / SR:.4f}s")
    if not (61.0 < len(amigo) / SR < 65.0):
        raise ValueError(f"Unexpected AMIGO duration: {len(amigo) / SR:.5f}s")

    out = np.zeros((n, CHANNELS), dtype=np.float32)

    # Entrance intro: existing v3 ocean ambience + a restrained, original
    # orchestral bed taken from v3 itself. Both fade into the AMIGO downbeat.
    intro_n = round(INTRO * SR)
    # Carry the ocean intro slightly across the 10s downbeat: a hard fade to
    # zero at exactly 10.0 would create an audible energy hole.
    intro_tail_n = round(0.65 * SR)
    intro_gate = np.ones(intro_n + intro_tail_n, dtype=np.float32)
    intro_gate[intro_n:] = np.cos(
        np.linspace(0, np.pi / 2.0, intro_tail_n, dtype=np.float32)
    )
    out[:intro_n + intro_tail_n] += (
        v3[:intro_n + intro_tail_n] * (1.05 * intro_gate[:, None])
    )
    motif = v3[round(11.0 * SR):round(21.0 * SR)]
    motif_n = min(intro_n, len(motif))
    t = np.arange(motif_n, dtype=np.float32) / SR
    fade_up = np.clip((t - 3.5) / 3.0, 0.0, 1.0)
    fade_down = np.clip((INTRO - t) / 0.95, 0.0, 1.0)
    out[:motif_n] += motif[:motif_n] * (0.62 * fade_up * fade_down)[:, None]
    # AMIGO remains at its original tempo / duration. Its measured input
    # loudness is much higher than v3, so -7.3dB is a starting *level match*,
    # not a claim that both tracks will have identical perceived loudness.
    amigo_end = round(INTRO * SR) + len(amigo)
    song = amigo.copy()
    song *= np.float32(10 ** (-7.3 / 20.0))
    fade_in_n = min(round(0.10 * SR), len(song))
    song[:fade_in_n] *= np.linspace(
        0, 1, fade_in_n, dtype=np.float32
    )[:, None]
    fade_out_start = round((STORM_START - INTRO) * SR)
    fade_out_start = max(0, min(len(song) - 1, fade_out_start))
    cross_n = len(song) - fade_out_start
    fade_progress = np.linspace(0.0, 1.0, cross_n, dtype=np.float32)
    song[fade_out_start:] *= np.cos(fade_progress * np.pi / 2.0)[:, None]
    out[round(INTRO * SR):amigo_end] += song

    # v3 re-enters on the *original absolute timeline* and reaches full
    # volume as AMIGO ends (at 1:12.57). From then through 3:00, all cues
    # including 1:29.2, 1:57, 2:30, 2:50, 2:58 retain their positions.
    storm_n = round(STORM_START * SR)
    ramp_n = max(1, amigo_end - storm_n)
    v3_gains = np.ones(n - storm_n, dtype=np.float32)
    v3_gains[:ramp_n] = np.sin(
        np.linspace(0, np.pi / 2.0, ramp_n, dtype=np.float32)
    )
    out[storm_n:] += v3[storm_n:] * v3_gains[:, None]

    return out


def master(pre: Path, dest: Path) -> dict:
    measured = loudness(pre)
    # Favor an uncompressed linear gain over hard dynamics processing.
    # Preserve v3's 2:50 chant duck and 2:58 ending.
    gain_db = min(
        TARGET_LUFS - measured["lufs"],
        TARGET_TP - measured["true_peak_dbtp"] - 0.12
    )
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(pre), "-af", f"volume={gain_db:.5f}dB",
        "-ar", str(SR), "-ac", str(CHANNELS), "-c:a", "pcm_s16le",
        str(dest)
    ])
    final = loudness(dest)
    if final["true_peak_dbtp"] > TARGET_TP + 0.08:
        raise RuntimeError(f"True peak exceeds target: {final}")
    if abs(duration(dest) - TOTAL) > 0.002:
        raise RuntimeError("Final duration differs from 180s")
    return {"before": measured, "applied_gain_db": round(gain_db, 3), "after": final}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amigo", type=Path, required=True,
                   help="Path to your legally obtained local AMIGO recording")
    p.add_argument("--v3-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    song = decode(a.amigo)
    print(f"AMIGO duration = {len(song)/SR:.5f}s", flush=True)
    report = {
        "source_amigo_duration_s": round(len(song)/SR, 5),
        "intro_s": INTRO,
        "amigo_start_s": INTRO,
        "amigo_end_s": round(INTRO + len(song)/SR, 5),
        "crossfade_start_s": STORM_START,
        "storm_full_s": round(INTRO + len(song)/SR, 5),
        "music_end_s": TOTAL,
        "master": {},
    }
    for label, filename in [
        ("music", "final_music_v3.wav"),
        ("with_voice", "final_music_with_voice_v3.wav"),
    ]:
        base = decode(a.v3_dir / filename)
        mixed = mix(song, base)
        pre = a.out_dir / f"fusion_D_{label}_pre.wav"
        final = a.out_dir / f"fusion_D_{label}_master.wav"
        save_wav(pre, mixed)
        report["master"][label] = master(pre, final)
        mp3 = a.out_dir / f"fusion_D_{label}_master.mp3"
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(final), "-c:a", "libmp3lame", "-b:a", "256k", str(mp3)])
        print(f"{label}: {report['master'][label]}", flush=True)
        print(f"output: {final}", flush=True)
        pre.unlink()
    (a.out_dir / "fusion_D_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(f"Done: {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
