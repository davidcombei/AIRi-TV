import gradio as gr
import shutil
import os
import subprocess

# ---- BASE PATHS (absolute) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_FOLDER = os.path.join(BASE_DIR, "assets")
VIDEO_FOLDER = os.path.join(ASSETS_FOLDER, "video")

IMAGE_EXT = [".png", ".jpg", ".jpeg"]
AUDIO_EXT = [".wav", ".mp3"]
PDF_EXT = [".pdf"]

def clean_assets():
    # Curăță fișiere din assets/
    for file in os.listdir(ASSETS_FOLDER):
        path = os.path.join(ASSETS_FOLDER, file)
        if os.path.isfile(path):
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXT or ext in AUDIO_EXT or ext in PDF_EXT:
                os.remove(path)

    # Curăță assets/video/
    if os.path.exists(VIDEO_FOLDER):
        for file in os.listdir(VIDEO_FOLDER):
            path = os.path.join(VIDEO_FOLDER, file)
            if os.path.isfile(path):
                os.remove(path)

    # Curăță assets/audio/
    audio_folder = os.path.join(ASSETS_FOLDER, "audio")
    if os.path.exists(audio_folder):
        for file in os.listdir(audio_folder):
            path = os.path.join(audio_folder, file)
            if os.path.isfile(path):
                os.remove(path)



def save_file(file_path, name):
    if file_path:
        ext = os.path.splitext(file_path)[1]
        destination = os.path.join(ASSETS_FOLDER, name + ext)
        shutil.copy(file_path, destination)


def upload_files(author_photo, author_audio, anchor_audio,
                 pdf_file, background_image, logo_image, anchor_image):

    clean_assets()

    save_file(anchor_image, "anchor")
    save_file(author_photo, "author_photo")
    save_file(background_image, "background")
    save_file(logo_image, "airi_logo")

    save_file(anchor_audio, "anchor")
    save_file(author_audio, "author_audio")

    save_file(pdf_file, "articol")

    return "✅ Files uploaded successfully!"


def start_generation(audience):

    command = [
        "./run_dissemination.sh",
        os.path.join(ASSETS_FOLDER, "articol.pdf"),
        os.path.join(ASSETS_FOLDER, "author_photo.png"),
        os.path.join(ASSETS_FOLDER, "author_audio.wav"),
        audience
    ]

    try:

        yield "🚀 Start generation...\n"

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output = "🚀 Start generation...\n"

        for line in iter(process.stdout.readline, ""):
            output += line
            yield output

        process.stdout.close()
        process.wait()

        yield output + "\n✅ Generation finished"

    except Exception as e:
        yield f"❌ Error running script: {str(e)}"


# -------- DOWNLOAD FUNCTIONS --------

def download_video():
    path = os.path.join(VIDEO_FOLDER, "articol_podcast_background.mp4")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, f"✅ Video ready for download!"
    else:
        return None, f"⏳ Video not generated"


def download_video_subtitles():
    path = os.path.join(VIDEO_FOLDER, "articol_podcast_background_subtitles.mp4")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, f"✅ Video + subtitles ready for download!"
    else:
        return None, f"⏳ Video + subtitles not generated"


def download_video_subtitles_images():
    path = os.path.join(VIDEO_FOLDER, "articol_podcast_background_subtitles_images.mp4")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, f"✅ Video + subtitles + images ready for download!"
    else:
        return None, f"⏳ Video + subtitles + images not generated"


# ---------------- UI ----------------

with gr.Blocks(title="AI Podcast Video Generator") as demo:

    gr.Markdown(
        """
        # 🎬 AI Podcast Video Generator
        Upload assets and generate your AI podcast video.
        """
    )

    with gr.Row():

        with gr.Column():

            gr.Markdown("## 🖼 Visual Assets")

            anchor_image = gr.Image(type="filepath", label="Anchor Photo")
            author_photo = gr.Image(type="filepath", label="Author Photo")
            background_image = gr.Image(type="filepath", label="Background Image")
            logo_image = gr.Image(type="filepath", label="Logo Image")

        with gr.Column():

            gr.Markdown("## 🔊 Audio")

            anchor_audio = gr.Audio(type="filepath", label="Anchor Audio")
            author_audio = gr.Audio(type="filepath", label="Author Audio")

            gr.Markdown("## 📄 Document")

            pdf_file = gr.File(label="PDF Document")

            gr.Markdown("## 🎯 Target Audience")

            audience = gr.Dropdown(
                choices=["liceeni", "firme", "politicieni"],
                label="Audience",
                value="liceeni"
            )

    gr.Markdown("---")

    upload_btn = gr.Button("🚀 Upload Assets", size="lg")
    upload_status = gr.Markdown()

    upload_btn.click(
        fn=upload_files,
        inputs=[
            author_photo,
            author_audio,
            anchor_audio,
            pdf_file,
            background_image,
            logo_image,
            anchor_image
        ],
        outputs=upload_status
    )

    gr.Markdown("---")

    start_btn = gr.Button("▶ Start Generation", size="lg")
    generation_status = gr.Textbox(
        label="Generation Logs",
        lines=25,
        autoscroll=True
    )

    start_btn.click(
        fn=start_generation,
        inputs=[audience],
        outputs=generation_status
    )

    # -------- DOWNLOAD SECTION --------

    gr.Markdown("## ⬇ Download Generated Videos")

    download_video_btn = gr.Button("Download Video")
    download_video_sub_btn = gr.Button("Download Video + Subtitles")
    download_video_full_btn = gr.Button("Download Video + Subtitles + Images")

    download_file = gr.File(label="Download File")
    download_status = gr.Markdown()

    download_video_btn.click(
        fn=download_video,
        outputs=[download_file, download_status]
    )

    download_video_sub_btn.click(
        fn=download_video_subtitles,
        outputs=[download_file, download_status]
    )

    download_video_full_btn.click(
        fn=download_video_subtitles_images,
        outputs=[download_file, download_status]
    )


demo.queue().launch(server_name="0.0.0.0", server_port=7860)

