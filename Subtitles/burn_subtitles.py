#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("Running:\n" + " ".join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main():
    ap = argparse.ArgumentParser(
        description="Burn-in (hardcode) SRT subtitles into a video, centered at the bottom."
    )
    ap.add_argument("input_video", help="Path to input video (mp4/mkv/etc.)")
    ap.add_argument("input_srt", help="Path to .srt file")
    ap.add_argument(
        "--out",
        default=None,
        help="Output video path (default: <input>_burned.mp4 in current dir)",
    )
    ap.add_argument(
        "--font-size",
        type=int,
        default=42,
        help="Subtitle font size (default: 24)",
    )
    ap.add_argument(
        "--bottom-margin",
        type=int,
        default=40,
        help="Bottom margin in pixels (default: 40)",
    )
    ap.add_argument(
        "--crf",
        type=int,
        default=18,
        help="x264 CRF quality (default: 18, lower=better)",
    )
    ap.add_argument(
        "--preset",
        default="veryfast",
        help="x264 preset (default: veryfast)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.input_video):
        print(f"ERROR: video not found: {args.input_video}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.input_srt):
        print(f"ERROR: srt not found: {args.input_srt}", file=sys.stderr)
        sys.exit(1)

    in_video = os.path.abspath(args.input_video)
    in_srt = os.path.abspath(args.input_srt)

    out_path = args.out
    if out_path is None:
        # ../assets/video relative to current working directory
        default_dir = Path.cwd() / "./assets/video"
        default_dir = default_dir.resolve()
        default_dir.mkdir(parents=True, exist_ok=True)

        out_path = default_dir / f"{Path(in_video).stem}_burned.mp4"

    out_path = os.path.abspath(out_path)

    # subtitles filter uses libass; style is ASS style override.
    # Alignment=2 -> bottom-center. MarginV controls distance from bottom.
    # ForceStyle values are ASS style keys.
    force_style = (
        f"Alignment=2,"
        f"FontSize={args.font_size},"
        f"MarginV={args.bottom_margin},"
        f"Outline=2,"
        f"Shadow=1"
    )

    # IMPORTANT: need to escape backslashes/colons on Windows; here we assume Linux paths.
    vf = f"subtitles={in_srt}:force_style='{force_style}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", in_video,
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", str(args.crf),
        "-preset", args.preset,
        "-c:a", "copy",
        out_path
    ]

    run(cmd)
    print(f"OK: wrote {out_path}")


if __name__ == "__main__":
    main()

