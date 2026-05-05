import argparse
import subprocess
import os
import sys

# 🔥 Path robust (independent de cwd)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "assets", "concatenare.mp4")


def run_command(cmd):
    print("Rulez:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Eroare la rularea comenzii!")
        sys.exit(1)


def convert_to_compatible(input_path, output_path):
    """
    Convertește video la format compatibil pentru concat:
    H.264 + AAC + 44100 Hz
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-ar", "44100",
        "-b:a", "128k",
        output_path
    ]
    run_command(cmd)


def concat_videos(video1, video2):
    temp1 = os.path.join(BASE_DIR, "temp_1.mp4")
    temp2 = os.path.join(BASE_DIR, "temp_2.mp4")

    # 🔧 asigură că folderul assets există
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 🔄 convertim ambele video-uri
    convert_to_compatible(video1, temp1)
    convert_to_compatible(video2, temp2)

    # 🔗 concat
    cmd = [
        "ffmpeg", "-y",
        "-i", temp1,
        "-i", temp2,
        "-filter_complex",
        "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]",
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        OUTPUT_PATH
    ]

    run_command(cmd)

    # 🧹 cleanup
    if os.path.exists(temp1):
        os.remove(temp1)
    if os.path.exists(temp2):
        os.remove(temp2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenează două video-uri")
    parser.add_argument("video1", help="Primul video (intro)")
    parser.add_argument("video2", help="Al doilea video (articol)")

    args = parser.parse_args()

    concat_videos(args.video1, args.video2)

    print(f"\n✅ Video final generat: {OUTPUT_PATH}")
