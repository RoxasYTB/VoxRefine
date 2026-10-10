#!/usr/bin/env python3
"""Compare isolated Cap60 CLI outputs sequentially and with two workers.

Inputs must be four frozen 16 kHz mono WAVs. Each inference gets a private
working directory and an identical 100 ms trailing guard. This benchmark does
not make dereverberation selection decisions.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

GUARD_SAMPLES = 1_600
SAMPLE_RATE = 16_000


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pcm_sha(audio: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(audio, dtype="<f4").tobytes()).hexdigest()


class GpuMemorySampler:
    def __init__(self, interval_s: float = .25) -> None:
        self.interval_s = interval_s
        self.values_mib: list[int] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                proc = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=2, check=False)
                if proc.returncode == 0:
                    self.values_mib.extend(int(line.strip()) for line in proc.stdout.splitlines()
                                           if line.strip())
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.stop_event.wait(self.interval_s)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)

    @property
    def peak_mib(self) -> int | None:
        return max(self.values_mib, default=None)


def run_one(executable: Path, source: Path, root: Path,
            thread_env: dict[str, str]) -> dict:
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE or audio.shape[1] != 1 or not np.isfinite(audio).all():
        raise ValueError(f"{source}: expected finite 16 kHz mono PCM")
    if len(audio) < SAMPLE_RATE // 2:
        raise ValueError(f"{source}: must contain at least 500 ms")
    job = root / source.stem
    input_dir, output_dir = job / "input", job / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir()
    guarded = np.concatenate((audio[:, 0], np.zeros(GUARD_SAMPLES, np.float32)))
    input_wav = input_dir / "input.wav"
    sf.write(input_wav, guarded, SAMPLE_RATE, subtype="PCM_16")
    env = os.environ.copy()
    env.update(thread_env)
    started = time.perf_counter()
    proc = subprocess.run([str(executable), "--atten-lim-db", "60",
        "--compensate-delay", "-o", str(output_dir), str(input_wav)],
        check=False, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - started
    if proc.returncode:
        raise RuntimeError(f"Cap60 failed for {source.name} ({proc.returncode}): "
            f"{proc.stderr[-2000:]} {proc.stdout[-1000:]}")
    outputs = list(output_dir.glob("*.wav"))
    if len(outputs) != 1:
        raise RuntimeError(f"expected exactly one Cap60 output for {source.name}, got {outputs}")
    result, out_sr = sf.read(outputs[0], dtype="float32", always_2d=True)
    if out_sr != SAMPLE_RATE or result.shape[1] != 1 or len(result) < len(audio):
        raise RuntimeError(f"invalid Cap60 result for {source.name}: sr={out_sr}, shape={result.shape}")
    trimmed = np.asarray(result[:len(audio), 0], dtype=np.float32)
    return {"input": source.name, "input_sha256": file_sha(source),
        "input_samples": int(len(audio)), "output_pcm_sha256": pcm_sha(trimmed),
        "output_samples": int(len(trimmed)), "elapsed_s": elapsed,
        "cap60_stdout_tail": proc.stdout[-600:], "audio": trimmed}


def run_sequential(executable: Path, inputs: list[Path], root: Path,
                   thread_env: dict[str, str]) -> tuple[dict[str, dict], float]:
    started = time.perf_counter()
    results = {}
    for index, source in enumerate(inputs):
        results[source.name] = run_one(executable, source, root / f"job-{index}", thread_env)
    return results, time.perf_counter() - started


def run_parallel(executable: Path, inputs: list[Path], root: Path,
                 thread_env: dict[str, str], order: list[int]) -> tuple[dict[str, dict], float]:
    started = time.perf_counter()
    results = {}
    # Two explicit waves avoid oversubscribing the 4 GB GPU.
    for wave_index, start in enumerate((0, 2)):
        batch = [(index, inputs[index]) for index in order[start:start + 2]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(run_one, executable, source,
                root / f"wave-{wave_index}-input-{index}", thread_env): source
                for index, source in batch}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results[result["input"]] = result
    return results, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-filter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2,
        help="OMP/MKL/OpenBLAS threads per isolated process")
    parser.add_argument("inputs", nargs=4, type=Path)
    args = parser.parse_args()
    executable = args.deep_filter.resolve()
    inputs = [path.resolve() for path in args.inputs]
    if not executable.is_file() or len({path.name for path in inputs}) != 4:
        raise ValueError("need an executable and four uniquely named input WAVs")
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError("all four frozen input WAVs must exist")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thread_env = {"OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads)}
    with GpuMemorySampler() as sampler_seq:
        sequential, seq_wall = run_sequential(executable, inputs,
            args.output_dir / "sequential", thread_env)
    peak_seq_mib = sampler_seq.peak_mib
    with GpuMemorySampler() as sampler_a:
        parallel_a, par_a_wall = run_parallel(executable, inputs,
            args.output_dir / "parallel-a", thread_env, [0, 1, 2, 3])
    peak_a_mib = sampler_a.peak_mib
    with GpuMemorySampler() as sampler_b:
        parallel_b, par_b_wall = run_parallel(executable, inputs,
            args.output_dir / "parallel-b", thread_env, [3, 2, 1, 0])
    peak_b_mib = sampler_b.peak_mib
    comparisons = []
    for source in inputs:
        expected = sequential[source.name]["audio"]
        for label, observed in (("parallel-a", parallel_a[source.name]),
                                ("parallel-b", parallel_b[source.name])):
            actual = observed["audio"]
            if actual.shape != expected.shape:
                max_abs = float("inf")
                identical = False
            else:
                max_abs = float(np.max(np.abs(actual - expected), initial=0.0))
                identical = bool(np.array_equal(actual, expected))
            comparisons.append({"input": source.name, "run": label,
                "sequential_pcm_sha256": sequential[source.name]["output_pcm_sha256"],
                "parallel_pcm_sha256": observed["output_pcm_sha256"],
                "pcm_bit_identical": identical, "max_abs_sample_error": max_abs})
    speedup = seq_wall / max(par_a_wall, par_b_wall)
    meets_exact_gate = all(x["pcm_bit_identical"] for x in comparisons)
    absolute_gate_s = 12.32
    parallel_wall_max = max(par_a_wall, par_b_wall)
    result = {"name": "Cap60-independent-process-concurrency-v1",
        "deep_filter_sha256": file_sha(executable),
        "input_sha256": {path.name: file_sha(path) for path in inputs},
        "thread_environment": thread_env, "worker_count": 2,
        "threads_per_worker": args.threads,
        "sequential_wall_s": seq_wall,
        "sequential_peak_gpu_memory_mib": peak_seq_mib,
        "parallel_a_wall_s": par_a_wall,
        "parallel_a_peak_gpu_memory_mib": peak_a_mib,
        "parallel_b_wall_s": par_b_wall,
        "parallel_b_peak_gpu_memory_mib": peak_b_mib,
        "parallel_wall_max_s": parallel_wall_max,
        "parallel_wall_absolute_gate_s": absolute_gate_s,
        "speedup_vs_sequential_using_slower_parallel_run": speedup,
        "pcm_comparisons": comparisons,
        "all_eight_pcm_comparisons_identical": meets_exact_gate,
        "all_max_sample_errors_zero": all(x["max_abs_sample_error"] == 0 for x in comparisons),
        "meets_two_worker_gate": (meets_exact_gate and speedup >= 1.5
            and parallel_wall_max <= absolute_gate_s),
        "test_wav_accessed": False}
    # Avoid writing sample arrays into the machine-readable report.
    for run in (sequential, parallel_a, parallel_b):
        for item in run.values():
            item.pop("audio", None)
    (args.output_dir / "report.json").write_text(json.dumps(result,
        indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items()
        if key not in ("input_sha256", "pcm_comparisons")}, indent=2))
    if not result["all_eight_pcm_comparisons_identical"]:
        raise SystemExit("Cap60 process concurrency is not PCM deterministic")


if __name__ == "__main__":
    main()
