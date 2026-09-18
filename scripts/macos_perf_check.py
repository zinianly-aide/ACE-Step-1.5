#!/usr/bin/env python3
"""Lightweight Apple Silicon preflight for ACE-Step.

This script does not generate audio. It reports the host configuration,
current swap/memory pressure, common competing processes, and the effective
ACE-Step performance profile so long 60-90s generations can be started with a
predictable memory footprint.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def run(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def bytes_to_gib(value: int) -> float:
    return value / (1024 ** 3)


def total_memory_gib() -> float | None:
    raw = run("sysctl", "-n", "hw.memsize")
    try:
        return bytes_to_gib(int(raw))
    except Exception:
        return None


def swap_summary() -> str:
    return run("sysctl", "vm.swapusage") or "unavailable"


def memory_pressure_summary() -> str:
    out = run("memory_pressure")
    if not out:
        return "unavailable"
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    preferred = [
        line for line in lines
        if "System-wide memory free percentage" in line
        or "Pages free" in line
        or "Pages purgeable" in line
    ]
    return " | ".join(preferred[:3]) or lines[-1]


def heavy_processes() -> list[str]:
    ps = run("ps", "-axo", "comm=,rss=")
    if not ps:
        return []
    patterns = (
        "ollama",
        "docker",
        "qemu",
        "unity",
        "android studio",
        "chrome",
        "localai",
    )
    found: list[tuple[int, str]] = []
    for line in ps.splitlines():
        match = re.match(r"\s*(.*?)\s+(\d+)\s*$", line)
        if not match:
            continue
        command, rss_kib = match.group(1), int(match.group(2))
        lower = command.lower()
        if any(p in lower for p in patterns):
            found.append((rss_kib, command))
    found.sort(reverse=True)
    return [f"{cmd} ({rss / 1024:.0f} MiB RSS)" for rss, cmd in found[:8]]


def api_health() -> str:
    host = os.getenv("ACESTEP_API_HOST", "127.0.0.1")
    port = os.getenv("ACESTEP_API_PORT", "8001")
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            return f"up ({resp.status})"
    except (urllib.error.URLError, TimeoutError, OSError):
        return "not running"


def show_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    print("ACE-Step macOS performance preflight")
    print("=" * 40)
    print(f"Host            : {platform.platform()}")
    print(f"Architecture    : {platform.machine()}")
    memory = total_memory_gib()
    print(f"Unified memory  : {memory:.1f} GiB" if memory is not None else "Unified memory  : unknown")
    print(f"Swap            : {swap_summary()}")
    print(f"Memory pressure : {memory_pressure_summary()}")
    print(f"API health      : {api_health()}")
    print()

    print("Effective M4 profile")
    print("-" * 40)
    profile = {
        "ACESTEP_LM_BACKEND": "mlx",
        "ACESTEP_LM_MODEL_PATH": "acestep-5Hz-lm-0.6B",
        "ACESTEP_USE_MLX_DIT": "false",
        "ACESTEP_MLX_VAE": "0",
        "ACESTEP_MLX_VAE_CHUNK": "192",
        "ACESTEP_OFFLOAD_TO_CPU": "false",
        "ACESTEP_OFFLOAD_DIT_TO_CPU": "false",
        "ACESTEP_QUEUE_WORKERS": "1",
        "ACESTEP_NO_INIT": "false",
    }
    for key, default in profile.items():
        print(f"{key:29} {show_env(key, default)}")
    print()

    heavy = heavy_processes()
    if heavy:
        print("Potential unified-memory competitors")
        print("-" * 40)
        for item in heavy:
            print(f"- {item}")
        print("For 60-90s generation, close unnecessary competitors before a long batch.")
        print()

    warnings: list[str] = []
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        warnings.append("This profile is intended for Apple Silicon macOS.")
    if show_env("ACESTEP_QUEUE_WORKERS", "1") != "1":
        warnings.append("ACESTEP_QUEUE_WORKERS should stay at 1 on 16GB unified memory.")
    if show_env("ACESTEP_USE_MLX_DIT", "false").lower() in {"1", "true", "yes", "on"}:
        warnings.append("MLX DiT is enabled; this can duplicate DiT residency and raise peak memory.")
    if show_env("ACESTEP_MLX_VAE", "0").lower() in {"1", "true", "yes", "on"}:
        warnings.append("MLX VAE is enabled; long audio decode may be less stable than tiled MPS on this profile.")

    if warnings:
        print("Warnings")
        print("-" * 40)
        for warning in warnings:
            print(f"- {warning}")
        return 1

    print(f"Profile looks consistent. Repo: {repo_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
