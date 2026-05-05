import argparse
import os
import json
import numpy as np
import torch
import open_clip
from PIL import Image


# ==============================
# SRT PARSER
# ==============================

def parse_srt(path):
    segments = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.isdigit():
            seg_id = int(line)

            if i + 1 >= len(lines):
                break

            time_line = lines[i + 1].strip()
            if "-->" not in time_line:
                i += 1
                continue

            start_str, end_str = time_line.split(" --> ")

            def srt_time_to_seconds(t):
                h, m, rest = t.split(":")
                s, ms = rest.split(",")
                return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

            start = srt_time_to_seconds(start_str)
            end = srt_time_to_seconds(end_str)

            text_lines = []
            j = i + 2
            while j < len(lines) and lines[j].strip() != "":
                text_lines.append(lines[j].strip())
                j += 1

            text = " ".join(text_lines)

            segments.append({
                "id": seg_id,
                "start": start,
                "end": end,
                "text": text
            })

            i = j
        else:
            i += 1

    return segments


# ==============================
# INTERVAL FINDER
# ==============================

def find_interval_for_image(similarities, segments, alpha=0.6):
    if len(similarities) == 0:
        return None

    peak_index = np.argmax(similarities)
    peak_score = similarities[peak_index]

    print(f"  Peak score: {peak_score:.4f}")

    threshold = alpha * peak_score
    print(f"  Threshold (alpha={alpha}): {threshold:.4f}")

    L = peak_index
    while L - 1 >= 0 and similarities[L - 1] >= threshold:
        L -= 1

    R = peak_index
    while R + 1 < len(similarities) and similarities[R + 1] >= threshold:
        R += 1

    print(f"  Interval indices: {L} -> {R}")

    included_segments = []
    for i in range(L, R + 1):
        included_segments.append({
            "id": segments[i]["id"],
            "start": segments[i]["start"],
            "end": segments[i]["end"],
            "text": segments[i]["text"]
        })

    return {
        "start": segments[L]["start"],
        "end": segments[R]["end"],
        "segments": included_segments
    }


# ==============================
# MAIN
# ==============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("srt_path", type=str)
    parser.add_argument("images_folder", type=str)
    args = parser.parse_args()

    print("=== Checking inputs ===")
    print("SRT path:", args.srt_path)
    print("Images folder:", args.images_folder)

    if not os.path.exists(args.srt_path):
        print("ERROR: SRT file not found")
        return

    if not os.path.isdir(args.images_folder):
        print("ERROR: Images folder not found")
        return

    print("\n=== Parsing SRT ===")
    segments = parse_srt(args.srt_path)
    print("Number of segments:", len(segments))

    if len(segments) == 0:
        print("ERROR: No segments parsed from SRT")
        return

    print("\n=== Loading model ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    print("\n=== Encoding text segments ===")
    texts = [seg["text"] for seg in segments]
    text_tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_features = text_features.cpu().numpy()

    image_files = [
        f for f in os.listdir(args.images_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print("\nImages found:", image_files)

    if len(image_files) == 0:
        print("ERROR: No images found")
        return

    results = []

    print("\n=== Processing images ===")

    for image_name in image_files:
        print(f"\nProcessing image: {image_name}")

        image_path = os.path.join(args.images_folder, image_name)

        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        image_features = image_features.cpu().numpy()[0]

        similarities = np.dot(text_features, image_features)

        print("  Max similarity:", np.max(similarities))

        interval = find_interval_for_image(similarities, segments, alpha=0.8)

        if interval is not None:
            results.append({
                "image": image_name,
                "interval": interval
            })

    output_path = os.path.join("assets", "pdf_images", "images_timestamps.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== DONE ===")
    print("Results count:", len(results))
    print("JSON saved at:", output_path)


if __name__ == "__main__":
    main()

