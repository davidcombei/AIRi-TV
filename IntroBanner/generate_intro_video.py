import argparse
import os
import subprocess
from moviepy.editor import ImageClip, vfx

DEFAULT_OUTPUT_DIR = "/workspace/AIRI-TV-TEST/AIRi-TV/assets"


def run_command(cmd):
    print("Rulez:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Eroare la FFmpeg")


def create_intro_video(input_image, output_video, duration=3.2):
    target_w, target_h = 1024, 512

    temp_video = output_video.replace(".mp4", "_no_audio.mp4")

    clip = ImageClip(input_image).set_duration(duration)

    # scale
    scale = max(target_w / clip.w, target_h / clip.h)
    clip = clip.resize(scale)

    # crop
    clip = clip.crop(
        x_center=clip.w / 2,
        y_center=clip.h / 2,
        width=target_w,
        height=target_h
    )

    # fade
    clip = clip.fx(vfx.fadein, 0.6).fx(vfx.fadeout, 0.6)

    # ❌ fără audio (evităm bug-ul)
    clip.write_videofile(
        temp_video,
        fps=25,
        codec="libx264",
        audio=False,
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    )

    # 🔥 adăugăm audio corect cu FFmpeg
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-f", "lavfi",
        "-t", str(duration),
        "-i", "anullsrc=channel_layout=mono:sample_rate=44100",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_video
    ]

    run_command(cmd)

    # 🧹 cleanup
    os.remove(temp_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image")
    parser.add_argument("--output")
    parser.add_argument("--duration", type=float, default=3.2)

    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, "intro.mp4")

    create_intro_video(args.input_image, output_path, args.duration)

    print(f"Video salvat la: {output_path}")
