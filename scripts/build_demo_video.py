from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1920, 1080
BACKGROUND = "#07111f"
PANEL = "#101e33"
TEXT = "#edf5ff"
MUTED = "#9eb1c8"
ACCENT = "#54d6c5"
ACCENT_2 = "#8b7cf6"

SLIDES: list[dict[str, Any]] = [
    {
        "eyebrow": "RAY + VLLM SYSTEMS LAB",
        "title": "One measured train-to-serve lifecycle",
        "bullets": [
            "Ray Data preprocessing",
            "Ray Train + PyTorch FSDP LoRA",
            "vLLM on Ray Serve",
            "Offline prefix-cache workload analysis",
        ],
        "narration": """This lab follows one small instruction model through the complete
systems lifecycle. It starts with distributed data preparation, fine-tunes a LoRA adapter with Ray
Train and PyTorch FSDP, restores durable checkpoints after an injected failure, and serves both the
base model and adapter through vLLM and Ray Serve. Every result shown here comes from checked-in raw
measurements.""",
    },
    {
        "eyebrow": "ARCHITECTURE",
        "title": "The same artifacts cross every boundary",
        "flow": [
            "JSONL\nrequests",
            "Ray Data\ntokenization",
            "Ray Train\nFSDP + LoRA",
            "Versioned\nadapter",
            "vLLM\nengine",
            "Ray Serve\nstreaming",
        ],
        "narration": """The architecture is intentionally integrated instead of a set of
disconnected notebooks. Revision-pinned JSONL data becomes tokenized Ray datasets. Ray Train
coordinates one or two FSDP workers using the same global batch and optimizer work. A validated
adapter checkpoint then moves into a single vLLM engine, where Ray Serve exposes a streaming
endpoint and the load generator records request and GPU telemetry.""",
    },
    {
        "eyebrow": "ORIGINAL SYSTEMS COMPONENT",
        "title": "Predict cacheability before allocating a GPU",
        "bullets": [
            "Exact chat-template tokenization",
            "Parent-linked hashes over complete KV blocks",
            "Block sizes 8, 16, and 32",
            "Arrival-order reuse with explicit namespaces",
        ],
        "images": ["charts/cacheability_by_block_size.png"],
        "narration": """The offline analyzer targets agentic traffic, where long system prompts,
tool definitions, and growing conversation histories create repeated token prefixes. It applies the
exact chat template, drops partial tail blocks, and builds a parent-linked hash chain over complete
blocks. The result is an arrival-order upper bound on reusable KV tokens for several block sizes,
without importing vLLM internals or loading model weights.""",
    },
    {
        "eyebrow": "DISTRIBUTED TRAINING",
        "title": "Small models expose coordination overhead",
        "bullets": [
            "1 GPU: 2,734.6 input tokens/s",
            "2 GPU FSDP: 1,256.5 input tokens/s",
            "Scaling efficiency: 22.97%",
            "Recovery: 21.78 s end to end; 2.52 s restore path",
            "Perplexity: 8.6308 → 7.5921",
        ],
        "images": [
            "charts/phase2/training_scaling.png",
            "charts/phase2/training_quality.png",
        ],
        "narration": """Training used the same Qwen zero point five billion model, fixed Dolly
split, global batch, and twelve optimizer steps on Modal L4 GPUs. One GPU processed two thousand
seven hundred thirty-five input tokens per second. Two-GPU FSDP reached only one thousand two
hundred fifty-seven, or twenty-three percent scaling efficiency, because collectives and
orchestration outweighed this tiny model's compute. The interrupted run recovered in twenty-one
point eight seconds, and held-out perplexity improved from eight point six three to seven point five
nine.""",
    },
    {
        "eyebrow": "PRODUCTION SERVING",
        "title": "1,920 streaming requests across 48 conditions",
        "bullets": [
            "Base model + rank-8 LoRA adapter",
            "Prefix cache off and on in isolated engines",
            "Concurrency 1, 8, 32, and 64",
            "Short, long, and repeated-prefix prompts",
            "Zero failed requests; 64 active streams observed",
        ],
        "images": [
            "charts/phase3/serving_throughput.png",
            "charts/phase3/serving_p95_latency.png",
        ],
        "narration": """Serving ran a full forty-eight-condition matrix: base and adapter targets,
cache off and on, three prompt shapes, and concurrency from one through sixty-four. A streaming
client measured time to first token, time per output token, p-fifty, p-ninety-five and p-ninety-nine
latency,
input and output throughput, GPU memory, utilization, cached tokens, and resource cost. All one
thousand nine hundred twenty requests completed, and Ray Serve observed sixty-four simultaneous
streams through one continuously batching engine.""",
    },
    {
        "eyebrow": "PREDICTION VS. OBSERVATION",
        "title": "Cache reuse is measurable—but not equal to speedup",
        "bullets": [
            "Prediction error: 0.28 pp mean; 0.95 pp maximum",
            "Agentic reuse at c=64: 94.59%",
            "Agentic throughput gain: only 5.7%–6.2%",
            "Long adapter throughput gain: 48.4%",
        ],
        "images": ["charts/phase3/cache_prediction.png"],
        "narration": """The key comparison is prediction versus observation. Across twenty-four
cache-on conditions, the offline full-block estimate had zero point two eight percentage points mean
absolute error and zero point nine five maximum error. But cache ratio is not a throughput forecast.
Agentic requests reused ninety-four point six percent of eligible block tokens at concurrency
sixty-four, yet throughput improved only about six percent because decode and scheduling remained.
Long adapter prompts gained forty-eight percent because shared prefill was a larger part of the
work.""",
    },
    {
        "eyebrow": "BOTTLENECKS AND FAILURES",
        "title": "The failures define the compatibility boundary",
        "bullets": [
            "FSDP adapter state was once filtered into an empty file",
            "Ray Serve ingress integration recursed in the full app",
            "vLLM 0.25.1 hit an unrelated MiniMax warm-up regression",
            "FlashInfer JIT expected a system CUDA toolkit",
            "Validated path: tensor checks, direct Starlette, vLLM 0.23",
        ],
        "narration": """The failure log is part of the result. An early FSDP checkpoint looked
valid but contained no adapter tensors after double filtering. The serving stack exposed a FastAPI
ingress serialization failure in the full application, an unrelated MiniMax warm-up regression in
vLLM zero point twenty-five, and a FlashInfer JIT path that expected a system CUDA toolkit. The
final path verifies every adapter tensor, uses Ray Serve's direct Starlette callable, pins a
validated vLLM
version, and selects the native sampler fallback.""",
    },
    {
        "eyebrow": "REPRODUCIBLE EVIDENCE",
        "title": "Raw traces → summaries → charts → reports",
        "bullets": [
            "make verify",
            "29 deterministic tests",
            "Revision-pinned model and dataset",
            "Artifact hashes and hardware manifests",
            "vLLM observability PR #53395",
        ],
        "narration": """To reproduce the CPU-side evidence, run make verify. GPU commands are
separated by phase and preserve model revisions, dependency versions, hardware, request traces, and
artifact hashes. The repository includes twenty-nine deterministic tests plus the raw inputs behind
every summary and chart. The measured cache study has also been shared with vLLM maintainers,
alongside a focused documentation pull request that explains how to measure automatic prefix-cache
effectiveness. That completes the train-to-serve systems lab.""",
    },
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSRounded.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default(size=size)


def _wrapped(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _place_image(canvas: Image.Image, source: Path, box: tuple[int, int, int, int]) -> None:
    chart = Image.open(source).convert("RGB")
    left, top, right, bottom = box
    chart.thumbnail((right - left - 40, bottom - top - 40), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (right - left, bottom - top), "#ffffff")
    x = (panel.width - chart.width) // 2
    y = (panel.height - chart.height) // 2
    panel.paste(chart, (x, y))
    canvas.paste(panel, (left, top))


def _render_slide(slide: dict[str, Any], index: int, output: Path) -> None:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((62, 50, 1858, 1030), radius=28, fill=PANEL)
    draw.rectangle((62, 50, 76, 1030), fill=ACCENT)
    draw.text((120, 92), slide["eyebrow"], fill=ACCENT, font=_font(28, bold=True))
    title_size = 46 if len(slide["title"]) > 43 else 54
    draw.text((120, 145), slide["title"], fill=TEXT, font=_font(title_size, bold=True))
    draw.text((1740, 94), f"{index + 1:02d} / {len(SLIDES):02d}", fill=MUTED, font=_font(25))

    images = slide.get("images", [])
    y = 300
    for bullet in slide.get("bullets", []):
        bullet_text = _wrapped(bullet, 38 if images else 75)
        bullet_font = _font(31 if images else 33)
        draw.ellipse((122, y + 11, 140, y + 29), fill=ACCENT_2)
        draw.multiline_text(
            (162, y),
            bullet_text,
            fill=TEXT,
            font=bullet_font,
            spacing=10,
        )
        bounds = draw.multiline_textbbox((162, y), bullet_text, font=bullet_font, spacing=10)
        y += bounds[3] - bounds[1] + 55

    flow = slide.get("flow")
    if flow:
        x, y = 120, 380
        box_width, gap = 252, 42
        for position, label in enumerate(flow):
            left = x + position * (box_width + gap)
            right = left + box_width
            draw.rounded_rectangle(
                (left, y, right, y + 190), radius=22, fill="#172944", outline=ACCENT_2, width=3
            )
            draw.multiline_text(
                (left + 22, y + 48), label, fill=TEXT, font=_font(31, bold=True), spacing=10
            )
            if position < len(flow) - 1:
                draw.line((right + 8, y + 95, right + gap - 8, y + 95), fill=ACCENT, width=5)
                draw.polygon(
                    [
                        (right + gap - 16, y + 84),
                        (right + gap - 4, y + 95),
                        (right + gap - 16, y + 106),
                    ],
                    fill=ACCENT,
                )

    if len(images) == 1:
        _place_image(canvas, REPOSITORY_ROOT / images[0], (850, 285, 1795, 895))
    elif len(images) == 2:
        _place_image(canvas, REPOSITORY_ROOT / images[0], (850, 270, 1795, 590))
        _place_image(canvas, REPOSITORY_ROOT / images[1], (850, 620, 1795, 940))

    draw.text(
        (120, 960),
        "Ray Data  •  Ray Train / FSDP  •  vLLM  •  Ray Serve",
        fill=MUTED,
        font=_font(24),
    )
    canvas.save(output)


def _run(*args: str) -> None:
    subprocess.run(args, check=True)


def _duration(path: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    )
    return float(output.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the narrated demonstration video.")
    parser.add_argument("--output", type=Path, default=Path("demo/ray-vllm-systems-lab-demo.mp4"))
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--rate", type=int, default=200)
    args = parser.parse_args()

    for executable in ("say", "ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable not found: {executable}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ray-vllm-demo-") as temporary:
        work = Path(temporary)
        segments: list[Path] = []
        for index, slide in enumerate(SLIDES):
            image_path = work / f"slide-{index:02d}.png"
            audio_path = work / f"audio-{index:02d}.aiff"
            video_path = work / f"segment-{index:02d}.mp4"
            _render_slide(slide, index, image_path)
            _run(
                "say",
                "-v",
                args.voice,
                "-r",
                str(args.rate),
                "-o",
                str(audio_path),
                slide["narration"],
            )
            segment_duration = _duration(audio_path) + 0.8
            _run(
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-i",
                str(image_path),
                "-i",
                str(audio_path),
                "-t",
                f"{segment_duration:.3f}",
                "-vf",
                "format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-tune",
                "stillimage",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-af",
                "apad",
                "-movflags",
                "+faststart",
                str(video_path),
            )
            segments.append(video_path)

        concat_path = work / "segments.txt"
        concat_path.write_text(
            "".join(f"file '{segment}'\n" for segment in segments),
            encoding="utf-8",
        )
        _run(
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(args.output),
        )

    print(f"wrote {args.output} ({_duration(args.output):.1f} seconds)")


if __name__ == "__main__":
    main()
