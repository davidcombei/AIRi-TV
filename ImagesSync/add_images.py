import cv2
import json
import os
import sys
import subprocess


def load_intervals(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    intervals = []
    for item in data:
        intervals.append({
            "image": item["image"],
            "start": item["interval"]["start"],
            "end": item["interval"]["end"]
        })

    return intervals


def get_active_image(intervals, current_time):
    for interval in intervals:
        if interval["start"] <= current_time <= interval["end"]:
            return interval["image"]
    return None


def process_video(video_path, json_path, image_folder):

    intervals = load_intervals(json_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Nu pot deschide video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = os.path.join("assets", "video")
    os.makedirs(output_dir, exist_ok=True)

    input_name = os.path.splitext(os.path.basename(video_path))[0]

    temp_video = os.path.join(output_dir, "temp_no_audio.mp4")
    final_video = os.path.join(output_dir, f"{input_name}_images.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    if not out.isOpened():
        print("EROARE: VideoWriter nu s-a deschis")
        return

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        image_name = get_active_image(intervals, current_time)

        if image_name:
            image_path = os.path.join(image_folder, image_name)

            if os.path.exists(image_path):
                img = cv2.imread(image_path)

                if img is not None:
                    
                    h, w = img.shape[:2]

                    y = 20

                    max_width = (2/3) * width
                    max_height = (height / 2) - y

                    scale_w = max_width / w
                    scale_h = max_height / h

                    scale = min(scale_w, scale_h)

                    new_w = int(w * scale)
                    new_h = int(h * scale)

                    img = cv2.resize(img, (new_w, new_h))

                    x = (width - new_w) // 2

                    frame[y:y+new_h, x:x+new_w] = img


        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print("Video fără audio creat.")

    # adaugă audio-ul original cu ffmpeg
    print("Adăugare audio cu ffmpeg...")

    command = [
        "ffmpeg",
        "-y",
        "-i", temp_video,
        "-i", video_path,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "copy",
        final_video
    ]

    subprocess.run(command)

    os.remove(temp_video)

    print("Video final salvat la:", os.path.abspath(final_video))


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python add_images.py video.mp4 data.json images_folder")
        sys.exit(1)

    video_path = sys.argv[1]
    json_path = sys.argv[2]
    image_folder = sys.argv[3]

    process_video(video_path, json_path, image_folder)
