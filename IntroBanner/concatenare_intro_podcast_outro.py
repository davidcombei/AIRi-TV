#!/usr/bin/env python3

import argparse
import subprocess
import sys


def run_command(cmd):
    print("Rulez:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Eroare la FFmpeg!")
        sys.exit(1)


def concat_videos(intro, podcast, outro, output):
    cmd = [
        "ffmpeg", "-y",
        "-i", intro,
        "-i", podcast,
        "-i", outro,
        "-filter_complex",
        "[0:v:0][0:a:0][1:v:0][1:a:0][2:v:0][2:a:0]concat=n=3:v=1:a=1[v][a]",
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "aac",
        "-ar", "44100",
        output
    ]

    run_command(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenează intro → podcast → outro"
    )
    parser.add_argument("intro", help="Calea către intro.mp4")
    parser.add_argument("podcast", help="Calea către video podcast")
    parser.add_argument("outro", help="Calea către outro.mp4")
    parser.add_argument(
        "--output",
        default="/workspace/AIRI-TV-TEST/AIRi-TV/assets/concatenare.mp4",
        help="Output final"
    )

    args = parser.parse_args()

    concat_videos(args.intro, args.podcast, args.outro, args.output)

    print(f"\n✅ Video final: {args.output}")
