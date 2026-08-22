from __future__ import annotations

import pytest

from ray_vllm_lab.serving.config import ServingConfig, config_from_dict


def test_request_counts_and_resource_cost() -> None:
    config = ServingConfig()

    assert config.request_count(1) == 32
    assert config.request_count(64) == 64
    assert config.request_count(1, smoke=True) == 8
    assert config.request_count(32, smoke=True) == 32
    assert config.requested_resource_hour_cost_usd == pytest.approx(1.4344)


def test_config_round_trip_normalizes_sequences() -> None:
    config = config_from_dict(ServingConfig().as_dict())

    assert config.concurrency_levels == (1, 8, 32, 64)
    config.validate()


def test_config_rejects_invalid_token_limits() -> None:
    with pytest.raises(ValueError, match="token limits"):
        ServingConfig(min_tokens=17, max_tokens=16).validate()
