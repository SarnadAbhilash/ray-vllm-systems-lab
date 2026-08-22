from __future__ import annotations

import json
from pathlib import Path

from ray_vllm_lab.training.metrics import build_scaling_summary


def test_checked_in_phase2_results_are_internally_consistent() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (repository_root / "results/phase2/raw/modal-results.json").read_text(encoding="utf-8")
    )

    rebuilt = build_scaling_summary(
        payload["one_gpu"]["metrics"],
        payload["two_gpu"]["metrics"],
        payload["recovery"]["metrics"],
    )
    assert rebuilt == payload["summary"]
    assert payload["recovery"]["metrics"]["resumed_from_checkpoint"] is True
    assert payload["recovery"]["metrics"]["replayed_optimizer_steps"] == 2
    assert len(payload["one_gpu"]["hardware"]["gpus"]) == 1
    assert len(payload["two_gpu"]["hardware"]["gpus"]) == 2
