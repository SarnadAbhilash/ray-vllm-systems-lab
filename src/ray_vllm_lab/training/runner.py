from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .config import TrainingConfig


class GpuSampler:
    """Sample device utilization and memory throughout the Ray Train fit call."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                result = subprocess.run(  # noqa: S603
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                sampled_at = time.time()
                for line in result.stdout.splitlines():
                    index, utilization, memory_mib = (
                        part.strip() for part in line.split(",", 2)
                    )
                    self.samples.append(
                        {
                            "sampled_at_unix_seconds": sampled_at,
                            "gpu_index": int(index),
                            "utilization_percent": float(utilization),
                            "memory_used_mib": float(memory_mib),
                        }
                    )
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, float | int]:
        utilization = [sample["utilization_percent"] for sample in self.samples]
        active = [value for value in utilization if value > 0]
        ordered = sorted(utilization)
        p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1) if ordered else 0
        return {
            "sample_count": len(self.samples),
            "mean_utilization_percent": (
                sum(utilization) / len(utilization) if utilization else 0.0
            ),
            "mean_active_utilization_percent": sum(active) / len(active) if active else 0.0,
            "p95_utilization_percent": ordered[p95_index] if ordered else 0.0,
            "peak_memory_used_gib": (
                max(sample["memory_used_mib"] for sample in self.samples) / 1024
                if self.samples
                else 0.0
            ),
        }


def _gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603
    inventory = []
    for line in result.stdout.splitlines():
        index, name, memory_mib, driver = (part.strip() for part in line.split(",", 3))
        inventory.append(
            {
                "index": int(index),
                "name": name,
                "memory_mib": int(memory_mib),
                "driver_version": driver,
            }
        )
    return inventory


def run_training_experiment(
    *,
    run_name: str,
    num_workers: int,
    inject_failure: bool,
    artifact_root: str = "/artifacts",
) -> dict[str, Any]:
    """Build Ray Data inputs, execute TorchTrainer, and persist a raw run manifest."""

    os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import ray
    import torch
    from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    from .data import build_ray_datasets
    from .train_loop import train_loop_per_worker

    config = TrainingConfig()
    config.validate(num_workers)
    artifact_path = Path(artifact_root)
    run_path = artifact_path / "phase2" / "runs" / run_name
    adapter_path = artifact_path / "phase2" / "adapters" / run_name
    marker_path = artifact_path / "phase2" / "failure-markers" / f"{run_name}.json"
    run_path.mkdir(parents=True, exist_ok=True)
    if marker_path.exists():
        marker_path.unlink()

    started_at = time.time()
    ray.init(
        num_gpus=num_workers,
        num_cpus=max(6, num_workers * 3),
        include_dashboard=False,
        log_to_driver=True,
    )
    try:
        datasets, preprocessing = build_ray_datasets(config)
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config={
                "training_config": config.as_dict(),
                "inject_failure": inject_failure,
                "failure_marker_path": str(marker_path),
            },
            datasets=datasets,
            scaling_config=ScalingConfig(
                num_workers=num_workers,
                use_gpu=True,
                resources_per_worker={"CPU": 2, "GPU": 1},
            ),
            run_config=RunConfig(
                name=run_name,
                storage_path=str(artifact_path / "phase2" / "ray-results"),
                failure_config=FailureConfig(max_failures=1 if inject_failure else 0),
                checkpoint_config=CheckpointConfig(num_to_keep=2),
            ),
        )
        sampler = GpuSampler()
        fit_started = time.time()
        sampler.start()
        try:
            result = trainer.fit()
        finally:
            sampler.stop()
        fit_seconds = time.time() - fit_started
        if result.checkpoint is None:
            raise RuntimeError("training completed without a final checkpoint")
        if adapter_path.exists():
            shutil.rmtree(adapter_path)
        result.checkpoint.to_directory(str(adapter_path))
        metrics = dict(result.metrics)
        metrics.pop("config", None)
        manifest = {
            "schema_version": "1.0",
            "run_name": run_name,
            "status": "complete",
            "num_workers": num_workers,
            "inject_failure": inject_failure,
            "started_at_unix_seconds": started_at,
            "finished_at_unix_seconds": time.time(),
            "fit_wall_seconds": fit_seconds,
            "config": config.as_dict(),
            "preprocessing": preprocessing,
            "metrics": metrics,
            "gpu_telemetry": {
                "scope": (
                    "entire Ray Train fit, including model load, evaluation, and checkpointing"
                ),
                "summary": sampler.summary(),
                "samples": sampler.samples,
            },
            "adapter_path": str(adapter_path),
            "hardware": {
                "gpus": _gpu_inventory(),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "ray": str(ray.__version__),
            },
        }
        (run_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return json.loads(json.dumps(manifest))
    finally:
        ray.shutdown()
