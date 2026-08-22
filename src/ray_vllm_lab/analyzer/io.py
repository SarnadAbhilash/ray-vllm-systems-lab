from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import RequestRecord


class InputFormatError(ValueError):
    """Raised when a JSONL row cannot be normalized."""


def _nonempty_string(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise InputFormatError(f"line {line_number}: {field} must be a non-empty string")
    return value


def _normalize_messages(value: Any, line_number: int) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise InputFormatError(f"line {line_number}: messages must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise InputFormatError(f"line {line_number}: messages[{index}] must be an object")
        role = _nonempty_string(message.get("role"), f"messages[{index}].role", line_number)
        content = message.get("content")
        if not isinstance(content, str):
            raise InputFormatError(
                f"line {line_number}: messages[{index}].content must be text; "
                "multimodal content is not supported in phase 1"
            )
        normalized.append({"role": role, "content": content})
    return tuple(normalized)


def normalize_row(row: dict[str, Any], line_number: int) -> RequestRecord:
    """Normalize plain, chat, or OpenAI Batch-style JSON into one record."""

    if not isinstance(row, dict):
        raise InputFormatError(f"line {line_number}: each JSONL row must be an object")

    body = row.get("body") if isinstance(row.get("body"), dict) else row
    request_id = str(
        row.get("custom_id")
        or row.get("id")
        or body.get("id")
        or f"request-{line_number:05d}"
    )
    adapter_id = body.get("adapter_id") or body.get("lora_id") or row.get("adapter_id")
    cache_salt = body.get("cache_salt") or row.get("cache_salt")

    if "messages" in body:
        return RequestRecord(
            request_id=request_id,
            messages=_normalize_messages(body["messages"], line_number),
            adapter_id=str(adapter_id) if adapter_id is not None else None,
            cache_salt=str(cache_salt) if cache_salt is not None else None,
            source_line=line_number,
        )
    if "prompt" in body:
        return RequestRecord(
            request_id=request_id,
            prompt=_nonempty_string(body["prompt"], "prompt", line_number),
            adapter_id=str(adapter_id) if adapter_id is not None else None,
            cache_salt=str(cache_salt) if cache_salt is not None else None,
            source_line=line_number,
        )
    raise InputFormatError(
        f"line {line_number}: expected prompt, messages, or an OpenAI Batch body containing one"
    )


def load_jsonl(path: str | Path) -> list[RequestRecord]:
    records: list[RequestRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise InputFormatError(f"line {line_number}: invalid JSON: {error.msg}") from error
            records.append(normalize_row(row, line_number))
    if not records:
        raise InputFormatError("input contains no requests")
    return records


def iter_jsonl(rows: Iterable[dict[str, Any]]) -> list[RequestRecord]:
    """Test and notebook helper for in-memory rows."""

    return [normalize_row(row, index) for index, row in enumerate(rows, start=1)]
