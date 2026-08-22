from ray_vllm_lab.analyzer.analysis import analyze_requests
from ray_vllm_lab.analyzer.models import TokenizedRequest


def request(
    request_id: str,
    tokens: list[int],
    *,
    adapter_id: str | None = None,
    cache_salt: str | None = None,
) -> TokenizedRequest:
    return TokenizedRequest(request_id, tuple(tokens), adapter_id, cache_salt)


def test_reuse_is_arrival_order_aware() -> None:
    report = analyze_requests(
        [
            request("first", list(range(8))),
            request("second", list(range(8)) + [20, 21, 22, 23]),
        ],
        model="test",
        block_sizes=(4,),
    )

    result = report.results[0]
    assert result.total_blocks == 5
    assert result.estimated_reusable_tokens == 8
    assert result.estimated_hit_ratio == 2 / 5
    assert result.shared_prefix_groups[0].shared_tokens == 8
    assert result.shared_prefix_groups[0].request_ids == ("first", "second")


def test_adapter_and_salt_partition_cache_identity() -> None:
    tokens = list(range(8))
    report = analyze_requests(
        [
            request("adapter-a", tokens, adapter_id="a"),
            request("adapter-b", tokens, adapter_id="b"),
            request("salted", tokens, adapter_id="a", cache_salt="tenant-7"),
        ],
        model="test",
        block_sizes=(4,),
    )

    assert report.results[0].estimated_reusable_tokens == 0
    assert report.results[0].shared_prefix_groups == ()


def test_partial_blocks_are_not_counted_as_reusable() -> None:
    report = analyze_requests(
        [request("a", list(range(10))), request("b", list(range(10)))],
        model="test",
        block_sizes=(4, 8),
    )

    four, eight = report.results
    assert four.full_block_tokens == 16
    assert four.tail_tokens == 4
    assert four.estimated_reusable_tokens == 8
    assert eight.full_block_tokens == 16
    assert eight.tail_tokens == 4
    assert eight.estimated_reusable_tokens == 8


def test_rejects_invalid_input() -> None:
    try:
        analyze_requests([], model="test")
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("expected ValueError")

