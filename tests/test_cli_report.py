import json
from typing import Any

from ray_vllm_lab.analyzer import cli
from ray_vllm_lab.analyzer.analysis import analyze_requests
from ray_vllm_lab.analyzer.models import TokenizedRequest
from ray_vllm_lab.analyzer.report import render_markdown, write_json, write_markdown


class ToyTokenizer:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def encode_prompt(self, prompt: str) -> list[int]:
        return [ord(character) for character in prompt]

    def encode_messages(self, messages: tuple[dict[str, Any], ...]) -> list[int]:
        text = "|".join(f"{item['role']}:{item['content']}" for item in messages)
        return self.encode_prompt(text)


def test_analyze_cli_writes_json_and_markdown(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "requests.jsonl"
    input_path.write_text(
        '{"id":"a","prompt":"shared-prefix-a"}\n'
        '{"id":"b","prompt":"shared-prefix-b"}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    monkeypatch.setattr(cli, "HuggingFaceConversationTokenizer", ToyTokenizer)

    exit_code = cli.main(
        [
            "analyze",
            str(input_path),
            "--model",
            "toy",
            "--block-sizes",
            "4",
            "--output",
            str(output_path),
            "--markdown",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text())["request_count"] == 2
    assert "Offline prefix-cache analysis" in markdown_path.read_text()
    assert "Reusable tokens" in capsys.readouterr().out


def test_compare_cli_writes_observed_result(tmp_path, capsys) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"results": [{"block_size": 16, "estimated_hit_ratio": 0.5}]})
    )
    metrics_path = tmp_path / "metrics.prom"
    metrics_path.write_text(
        "vllm:prefix_cache_queries 100\nvllm:prefix_cache_hits 40\n", encoding="utf-8"
    )
    output_path = tmp_path / "comparison.json"

    assert (
        cli.main(
            [
                "compare",
                str(report_path),
                "--metrics",
                str(metrics_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert json.loads(output_path.read_text())["observed_hit_ratio"] == 0.4
    assert "absolute_error" in capsys.readouterr().out


def test_report_helpers_render_and_write(tmp_path) -> None:
    report = analyze_requests(
        [
            TokenizedRequest("a", tuple(range(8))),
            TokenizedRequest("b", tuple(range(8))),
        ],
        model="toy",
        block_sizes=(4,),
    )
    json_path = tmp_path / "nested" / "report.json"
    markdown_path = tmp_path / "nested" / "report.md"

    write_json(report, json_path)
    write_markdown(report, markdown_path)

    assert json.loads(json_path.read_text())["model"] == "toy"
    rendered = render_markdown(report)
    assert "50.0%" in rendered
    assert markdown_path.read_text() == rendered

