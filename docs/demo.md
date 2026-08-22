# Three-minute demonstration

The narrated demonstration is available at
[`demo/ray-vllm-systems-lab-demo.mp4`](../demo/ray-vllm-systems-lab-demo.mp4).
It walks through the integrated architecture, offline analyzer, distributed training and recovery,
serving matrix, predicted-versus-observed cache behavior, bottleneck investigation, and retained
integration failures.

## Rebuild the video

The builder composes the checked-in charts into 1080p result cards, generates narration with the
macOS system voice, and encodes the final H.264/AAC video with FFmpeg:

```bash
uv sync --extra charts
uv run python scripts/build_demo_video.py
```

Use `--voice` and `--rate` to change the narration while preserving the visuals. The complete spoken
script is stored in the `SLIDES` data inside the builder so that narration and result cards remain in
sync.
