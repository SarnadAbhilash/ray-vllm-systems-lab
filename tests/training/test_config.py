import pytest

from ray_vllm_lab.training.config import TrainingConfig, config_from_dict


def test_gradient_accumulation_keeps_global_batch_constant() -> None:
    config = TrainingConfig()

    assert config.gradient_accumulation_steps(1) == 2
    assert config.gradient_accumulation_steps(2) == 1
    assert config.micro_batch_size * 1 * config.gradient_accumulation_steps(1) == 8
    assert config.micro_batch_size * 2 * config.gradient_accumulation_steps(2) == 8


def test_config_round_trip_preserves_tuple() -> None:
    config = TrainingConfig()
    restored = config_from_dict(config.as_dict())

    assert restored == config
    assert isinstance(restored.target_modules, tuple)


def test_invalid_experiment_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="one or two"):
        TrainingConfig().validate(3)
    with pytest.raises(ValueError, match="train_examples"):
        TrainingConfig(train_examples=8).validate(1)
    with pytest.raises(ValueError, match="failure_after_step"):
        TrainingConfig(failure_after_step=6).validate(1)
