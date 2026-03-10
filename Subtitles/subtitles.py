#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile
import whisper


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")
    return p


def format_time(seconds: float) -> str:
    # round to nearest millisecond; normalize overflow
    ms_total = int(round(seconds * 1000.0))
    if ms_total < 0:
        ms_total = 0

    hours = ms_total // 3_600_000
    ms_total %= 3_600_000
    minutes = ms_total // 60_000
    ms_total %= 60_000
    secs = ms_total // 1000
    millis = ms_total % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def extract_audio_wav(video_path: str, wav_path: str):
    # Extract EXACT audio timeline from the final video.
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        wav_path
    ]
    run(cmd)


def generate_srt_from_wav(wav_path: str, out_srt: str, model_name: str, language: str | None):
    model = whisper.load_model(model_name)

    kwargs = {}
    if language:
        kwargs["language"] = language
        kwargs["task"] = "transcribe"

    result = model.transcribe(wav_path, **kwargs)

    segments = result.get("segments", [])
    if not segments:
        raise RuntimeError("No segments returned by Whisper.")

    os.makedirs(os.path.dirname(out_srt), exist_ok=True)

    with open(out_srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = float(seg["start"])
            end = float(seg["end"])
            text = (seg.get("text") or "").strip()

            if end < start:
                end = start

            f.write(f"{i}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text}\n\n")


def main():
    default_out = os.path.normpath(os.path.join(os.getcwd(), "./assets/subtitles/subtitles.srt"))

    ap = argparse.ArgumentParser(
        description="Generate drift-free SRT by extracting audio from the FINAL video, then transcribing with Whisper."
    )
    ap.add_argument("input_video", help="Path to FINAL rendered video (the one you will watch)")
    ap.add_argument("output_srt", nargs="?", default=default_out,
                    help=f"Path to output .srt (default: {default_out})")
    ap.add_argument("--model", default="large-v3", help="Whisper model name (default: large-v3)")
    ap.add_argument("--language", default=None, help="Language code (e.g., ro, en). If set, disables auto-detect.")
    args = ap.parse_args()

    if not os.path.isfile(args.input_video):
        print(f"ERROR: video not found: {args.input_video}", file=sys.stderr)
        sys.exit(1)

    # Make output path absolute & normalized
    out_srt = os.path.abspath(args.output_srt)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "extracted.wav")
        extract_audio_wav(args.input_video, wav_path)
        generate_srt_from_wav(wav_path, out_srt, args.model, args.language)

    print(f"OK: wrote {out_srt}")


if __name__ == "__main__":
    main()

