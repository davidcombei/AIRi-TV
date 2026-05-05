#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=4

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <article.pdf> <author_image.png> <author_reference.wav> <audience> <title> <author_name>"
    exit 1
fi

ARTICLE="$1"
AUTHOR_IMG="$2"
AUTHOR_REF_AUDIO="$3"
AUDIENCE="$4"
TITLE="$5"
AUTHOR_NAME="$6"
ANCHOR_REF_AUDIO="assets/anchor.wav"
MEASURE="python Measurements/GPU_CONSUMPTION/measure.py"

# Generam bannerul :
$MEASURE "IntroBanner" \
    conda run -n intro_banner python IntroBanner/generate_intro_banner.py \
        "$TITLE" \
        "$AUTHOR_NAME" \
        "$AUTHOR_IMG"

# Generam videoul de intro :
$MEASURE "IntroVideo" \
    conda run -n intro_banner python IntroBanner/generate_intro_video.py \
        "assets/banner_output.jpeg"

# Background removal :
$MEASURE "BackgroundRemoval" \
    conda run -n backgrounds python3 remove_background.py \
        "assets/anchor.jpg" \
        "assets/"

conda run -n backgrounds python3 remove_background.py \
    "$AUTHOR_IMG" \
    "assets/"

ANCHOR_IMG="assets/anchor.png"
AUTHOR_BASENAME="${AUTHOR_IMG##*/}"
AUTHOR_NAME="${AUTHOR_BASENAME%.*}"
AUTHOR_IMG="assets/${AUTHOR_NAME}.png"

# LLM :
$MEASURE "LLM" \
    conda run -n llama python3 LLM/llama_audience.py \
        "$ARTICLE" \
        "assets/text/" \
        "$AUDIENCE"
echo "LLM done"

# TTS :
$MEASURE "TTS_Chatterbox" \
    conda run -n chatterbox python3 TTS/chatterbox/run_chatterbox.py \
        "$AUTHOR_REF_AUDIO" \
        "$ANCHOR_REF_AUDIO" \
        "assets/text/conversation.txt" \
        "assets/audio/"
echo "audios done"

$MEASURE "TTS_Concatenate" \
    conda run -n chatterbox python3 TTS/chatterbox/concatenate_audios.py \
        "assets/audio/"
echo "concatenation done"

# SadTalker Anchor :
$MEASURE "SadTalker_Anchor" \
    conda run -n sadtalker python3 VisualModel/SadTalker/inference.py \
        --driven_audio assets/audio/anchor_full_audio.wav \
        --source_image $ANCHOR_IMG \
        --enhancer gfpgan \
        --result_dir assets/video \
        --checkpoint_dir VisualModel/SadTalker/checkpoints

ANCHOR_VIDEO=$(ls -t assets/video/*.mp4 | head -n 1)
echo "ANCHOR VIDEO = $ANCHOR_VIDEO"

# SadTalker Author :
$MEASURE "SadTalker_Author" \
    conda run -n sadtalker python3 VisualModel/SadTalker/inference.py \
        --driven_audio assets/audio/author_full_audio.wav \
        --source_image $AUTHOR_IMG \
        --enhancer gfpgan \
        --result_dir assets/video/ \
        --checkpoint_dir VisualModel/SadTalker/checkpoints

AUTHOR_VIDEO=$(ls -t assets/video/*.mp4 | head -n 1)
echo "AUTHOR VIDEO = $AUTHOR_VIDEO"

ARTICLE_BASENAME="${ARTICLE##*/}"
ARTICLE_NAME="${ARTICLE_BASENAME%.*}"

# FFmpeg merge :
$MEASURE "FFmpeg_Merge" \
    ffmpeg \
        -i "$ANCHOR_VIDEO" \
        -i "$AUTHOR_VIDEO" \
        -filter_complex "\
            [0:v]setpts=PTS-STARTPTS[left]; \
            [1:v]setpts=PTS-STARTPTS[right]; \
            [left][right]hstack=inputs=2[v]; \
            [0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[a]" \
        -map "[v]" \
        -map "[a]" \
        -c:v libx264 -preset veryfast -crf 18 \
        -c:a aac -b:a 192k \
        "assets/video/${ARTICLE_NAME}_podcast.mp4"
echo "merging done"

# Background + logo :
echo "adding background and logo to video"
$MEASURE "BackgroundVideo" \
    conda run -n backgrounds python3 add_background_video.py \
        "assets/video/${ARTICLE_NAME}_podcast.mp4" \
        "assets/video/${ARTICLE_NAME}_podcast_background.mp4" \
        "assets/background.jpg" \
        "assets/airi_logo.jpeg"

# Subtitles :
$MEASURE "Subtitles_Generate" \
    conda run -n subtitles python3 Subtitles/subtitles.py \
        "assets/video/${ARTICLE_NAME}_podcast_background.mp4"

$MEASURE "Subtitles_Burn" \
    conda run -n subtitles python3 Subtitles/burn_subtitles.py \
        "assets/video/${ARTICLE_NAME}_podcast_background.mp4" \
        "assets/subtitles/subtitles.srt" \
        --font-size 24 \
        --bottom-margin 20

# Images sync :
$MEASURE "ImagesSync_Extract" \
    conda run -n pdf_images python ImagesSync/extract_pdf_images.py \
        "assets/${ARTICLE_NAME}.pdf"
echo "extracting images done"

$MEASURE "ImagesSync_Timestamps" \
    conda run -n pdf_images python3 ImagesSync/create_images_timestamps.py \
        assets/subtitles/subtitles.srt \
        assets/pdf_images
echo "creating images timestamps done"

$MEASURE "ImagesSync_Add" \
    conda run -n pdf_images python ImagesSync/add_images.py \
        assets/video/"${ARTICLE_NAME}_podcast_background_subtitles.mp4" \
        assets/pdf_images/images_timestamps.json \
        assets/pdf_images
echo "video rendered with images"

echo "videos done, all being saved to ./assets/video"
echo "GPU measurements saved to Measurements/GPU_CONSUMPTION/RESULTS/"
