import json

import pytest

from ray_vllm_lab.analyzer.metrics import compare_report, parse_prometheus


def test_parses_and_sums_labeled_prometheus_samples() -> None:
    metrics = parse_prometheus(
        """
        # HELP vllm:prefix_cache_hits Prefix hits
        vllm:prefix_cache_hits{model_name="base"} 40
        vllm:prefix_cache_hits{model_name="adapter"} 10
        vllm:prefix_cache_queries{model_name="base"} 100
        """
    )

    assert metrics["vllm:prefix_cache_hits"] == 50
    assert metrics["vllm:prefix_cache_queries"] == 100


def test_compares_prediction_with_observation(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"results": [{"block_size": 16, "estimated_hit_ratio": 0.75}]})
    )
    comparison = compare_report(
        report_path,
        {"vllm:prefix_cache_hits": 60, "vllm:prefix_cache_queries": 100},
        16,
    )

    assert comparison["observed_hit_ratio"] == 0.6
    assert comparison["absolute_error"] == pytest.approx(0.15)

