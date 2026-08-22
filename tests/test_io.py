import pytest

from ray_vllm_lab.analyzer.io import InputFormatError, iter_jsonl, load_jsonl


def test_normalizes_plain_chat_and_batch_rows() -> None:
    records = iter_jsonl(
        [
            {"id": "plain", "prompt": "hello"},
            {"id": "chat", "messages": [{"role": "user", "content": "hello"}]},
            {
                "custom_id": "batch",
                "body": {
                    "model": "demo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "lora_id": "support-v1",
                },
            },
        ]
    )

    assert [record.request_id for record in records] == ["plain", "chat", "batch"]
    assert records[0].prompt == "hello"
    assert records[1].messages[0]["role"] == "user"
    assert records[2].adapter_id == "support-v1"


def test_rejects_multimodal_message_content() -> None:
    with pytest.raises(InputFormatError, match="multimodal"):
        iter_jsonl(
            [
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
                    ]
                }
            ]
        )


def test_load_jsonl_rejects_invalid_or_empty_input(tmp_path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(InputFormatError, match="invalid JSON"):
        load_jsonl(invalid)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(InputFormatError, match="no requests"):
        load_jsonl(empty)
