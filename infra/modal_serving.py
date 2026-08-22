from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DATA_ROOT = REPOSITORY_ROOT / "data"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

app = modal.App("ray-vllm-systems-lab-serving")
artifact_volume = modal.Volume.from_name("ray-vllm-lab-artifacts", create_if_missing=True)
cache_volume = modal.Volume.from_name("ray-vllm-lab-cache", create_if_missing=True)

serving_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(REPOSITORY_ROOT / "infra/requirements-serving.txt"))
    .add_local_dir(str(SOURCE_ROOT), remote_path="/root/src", copy=True)
    .add_local_dir(str(DATA_ROOT), remote_path="/root/data", copy=True)
    .env(
        {
            "PYTHONPATH": "/root/src",
            "HF_HOME": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "RAY_USAGE_STATS_ENABLED": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
)

FUNCTION_OPTIONS = {
    "image": serving_image,
    "gpu": "L4",
    "cpu": 8,
    "memory": 32768,
    "timeout": 3600,
    "volumes": {"/artifacts": artifact_volume, "/cache": cache_volume},
    "max_containers": 1,
}


def _remote_run(run_name: str, prefix_caching: bool, smoke: bool) -> dict[str, Any]:
    from ray_vllm_lab.serving.runner import run_serving_experiment

    result = run_serving_experiment(
        run_name=run_name,
        prefix_caching=prefix_caching,
        smoke=smoke,
    )
    artifact_volume.commit()
    cache_volume.commit()
    return result


@app.function(**FUNCTION_OPTIONS)
def benchmark_prefix_off(run_name: str, smoke: bool = False) -> dict[str, Any]:
    return _remote_run(run_name, prefix_caching=False, smoke=smoke)


@app.function(**FUNCTION_OPTIONS)
def benchmark_prefix_on(run_name: str, smoke: bool = False) -> dict[str, Any]:
    return _remote_run(run_name, prefix_caching=True, smoke=smoke)


def _compact(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: manifest[key]
        for key in (
            "run_name",
            "prefix_caching",
            "smoke",
            "startup_seconds",
            "benchmark_seconds",
            "config",
            "workloads",
            "conditions",
            "continuous_batching",
            "gpu_telemetry",
            "hardware",
        )
    }


@app.local_entrypoint()
def main(
    experiment_id: str = "",
    output: str = "results/phase3/raw/modal-results.json",
    smoke: bool = False,
) -> None:
    """Run cache-off and cache-on modes sequentially for clean GPU isolation."""

    experiment_id = experiment_id or datetime.now(UTC).strftime("phase3-%Y%m%d-%H%M%S")
    prefix_off = benchmark_prefix_off.remote(f"{experiment_id}-prefix-off", smoke)
    prefix_on = benchmark_prefix_on.remote(f"{experiment_id}-prefix-on", smoke)
    aggregate = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "prefix_off": _compact(prefix_off),
        "prefix_on": _compact(prefix_on),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "prefix_off_conditions": len(prefix_off["conditions"]),
                "prefix_on_conditions": len(prefix_on["conditions"]),
                "output": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
