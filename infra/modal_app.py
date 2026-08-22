from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

app = modal.App("ray-vllm-systems-lab-training")
artifact_volume = modal.Volume.from_name("ray-vllm-lab-artifacts", create_if_missing=True)
cache_volume = modal.Volume.from_name("ray-vllm-lab-cache", create_if_missing=True)

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(REPOSITORY_ROOT / "infra/requirements-training.txt"))
    .add_local_dir(str(SOURCE_ROOT), remote_path="/root/src", copy=True)
    .env(
        {
            "PYTHONPATH": "/root/src",
            "HF_HOME": "/cache/huggingface",
            "RAY_TRAIN_V2_ENABLED": "1",
            "RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

FUNCTION_OPTIONS = {
    "image": training_image,
    "cpu": 8,
    "memory": 32768,
    "timeout": 3600,
    "volumes": {"/artifacts": artifact_volume, "/cache": cache_volume},
    "max_containers": 1,
}


def _remote_run(run_name: str, num_workers: int, inject_failure: bool) -> dict[str, Any]:
    from ray_vllm_lab.training.runner import run_training_experiment

    result = run_training_experiment(
        run_name=run_name,
        num_workers=num_workers,
        inject_failure=inject_failure,
    )
    artifact_volume.commit()
    cache_volume.commit()
    return result


@app.function(gpu="L4", **FUNCTION_OPTIONS)
def train_one_gpu(run_name: str) -> dict[str, Any]:
    return _remote_run(run_name, num_workers=1, inject_failure=False)


@app.function(gpu="L4:2", **FUNCTION_OPTIONS)
def train_two_gpu(run_name: str) -> dict[str, Any]:
    return _remote_run(run_name, num_workers=2, inject_failure=False)


@app.function(gpu="L4:2", **FUNCTION_OPTIONS)
def train_recovery_probe(run_name: str) -> dict[str, Any]:
    return _remote_run(run_name, num_workers=2, inject_failure=True)


def _compact(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": manifest["run_name"],
        "num_workers": manifest["num_workers"],
        "inject_failure": manifest["inject_failure"],
        "metrics": manifest["metrics"],
        "preprocessing": manifest["preprocessing"],
        "gpu_telemetry": manifest["gpu_telemetry"],
        "hardware": manifest["hardware"],
        "adapter_path": manifest["adapter_path"],
    }


@app.local_entrypoint()
def main(
    experiment_id: str = "",
    output: str = "results/phase2/raw/modal-results.json",
) -> None:
    """Run the full Phase-2 matrix sequentially to avoid hidden resource overlap."""

    experiment_id = experiment_id or datetime.now(UTC).strftime("phase2-%Y%m%d-%H%M%S")
    one_gpu = train_one_gpu.remote(f"{experiment_id}-1gpu")
    two_gpu = train_two_gpu.remote(f"{experiment_id}-2gpu")
    recovery = train_recovery_probe.remote(f"{experiment_id}-recovery")

    from ray_vllm_lab.training.metrics import build_scaling_summary

    aggregate = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "one_gpu": _compact(one_gpu),
        "two_gpu": _compact(two_gpu),
        "recovery": _compact(recovery),
        "summary": build_scaling_summary(
            one_gpu["metrics"],
            two_gpu["metrics"],
            recovery["metrics"],
        ),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate["summary"], indent=2, sort_keys=True))
