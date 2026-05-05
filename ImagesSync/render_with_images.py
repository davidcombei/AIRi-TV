#!/usr/bin/env python3
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def ffprobe_fps(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "json",
        video_path
    ]
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)

    data = json.loads(p.stdout)
    stream = data["streams"][0]

    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    num, den = rate.split("/")
    fps = float(num) / float(den)
    return fps


def load_items(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for obj in data:
        img = obj.get("image")
        interval = obj.get("interval", {})
        if not img:
            continue
        start = interval.get("start")
        end = interval.get("end")
        if start is None or end is None:
            continue
        if float(end) <= float(start):
            continue

        items.append({
            "image": img,
            "start": float(start),
            "end": float(end)
        })

    return items


def resolve_overlaps_keep_first(items):
    # sortăm după start; prima câștigă
    items = sorted(items, key=lambda x: x["start"])
    result = []
    last_end = -1

    for item in items:
        if item["start"] >= last_end:
            result.append(item)
            last_end = item["end"]
        # dacă se suprapune → ignorăm

    return result


def build_filter(items, fps):
    parts = []
    parts.append(f"[0:v]fps={fps},format=rgba[base0]")

    for idx, item in enumerate(items):
        img_index = idx + 1

        start_frame = math.ceil(item["start"] * fps)
        end_frame = math.ceil(item["end"] * fps) - 1

        parts.append(f"[{img_index}:v]format=rgba[img{idx}]")

        # scalare la max 50% din lățimea video
        parts.append(
            f"[img{idx}][base{idx}]scale2ref="
            f"w='min(iw,main_w*0.5)':h=-1"
            f"[imgS{idx}][baseA{idx}]"
        )

        # overlay jos-centru, 20px margine
        parts.append(
            f"[baseA{idx}][imgS{idx}]overlay="
            f"x='(W-w)/2':y='H-h-20':"
            f"enable='between(n,{start_frame},{end_frame})'"
            f"[base{idx+1}]"
        )

    final_label = f"[base{len(items)}]"
    parts.append(f"{final_label}format=yuv420p[vout]")

    return ";".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Render video with images using JSON intervals (frame accurate)"
    )
    parser.add_argument("json_path")
    parser.add_argument("images_dir")
    parser.add_argument("video_path")

    args = parser.parse_args()

    if not os.path.isfile(args.json_path):
        print("JSON not found")
        sys.exit(1)

    if not os.path.isdir(args.images_dir):
        print("Images folder not found")
        sys.exit(1)

    if not os.path.isfile(args.video_path):
        print("Video not found")
        sys.exit(1)

    fps = ffprobe_fps(args.video_path)

    items = load_items(args.json_path)
    items = resolve_overlaps_keep_first(items)

    valid_items = []
    for item in items:
        img_path = os.path.join(args.images_dir, item["image"])
        if os.path.isfile(img_path):
            item["image_path"] = img_path
            valid_items.append(item)

    if not valid_items:
        print("No valid images found.")
        sys.exit(1)

    filter_complex = build_filter(valid_items, fps)

    output_path = str(Path.cwd() / f"{Path(args.video_path).stem}_rendered.mp4")

    cmd = ["ffmpeg", "-y", "-i", args.video_path]

    for item in valid_items:
        cmd += ["-loop", "1", "-i", item["image_path"]]

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-r", str(fps),
        "-fps_mode", "cfr",
        "-c:a", "copy",
        output_path
    ]

    print("Running command:")
    print(" ".join(shlex.quote(c) for c in cmd))

    subprocess.run(cmd)

    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()

