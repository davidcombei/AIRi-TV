#!/bin/bash
set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <article.pdf> <author_image.png> <author_reference.wav>"
    exit 1
fi

ARTICLE="$1"
AUTHOR_IMG="$2"
AUTHOR_REF_AUDIO="$3"


ANCHOR_IMG="assets/anchor.png"
ANCHOR_REF_AUDIO="assets/LJ001-0001.wav"

echo



conda run -n llama python3 LLM/llama_3-8B.py \
    "$ARTICLE" \
    "assets/text/"

echo "LLM done"




conda run -n chatterbox python3 TTS/chatterbox/run_chatterbox.py \
    "$AUTHOR_REF_AUDIO" \
    "$ANCHOR_REF_AUDIO" \
    "assets/text/conversation.txt" \
    "assets/audio/"


echo "audios done"

conda run -n chatterbox python3 TTS/chatterbox/concatenate_audios.py \
    "assets/audio/"

echo "concatenation done"
 


conda run -n sadtalker python3 VisualModel/SadTalker/inference.py \
    --driven_audio assets/audio/anchor_full_audio.wav \
    --source_image $ANCHOR_IMG \
    --enhancer gfpgan \
    --result_dir assets/video \
    --checkpoint_dir VisualModel/SadTalker/checkpoints \

ANCHOR_VIDEO=$(ls -t assets/video/*.mp4 | head -n 1)
echo "ANCHOR VIDEO = $ANCHOR_VIDEO

conda run -n sadtalker python3 VisualModel/SadTalker/inference.py \
    --driven_audio assets/audio/author_full_audio.wav \
    --source_image $AUTHOR_IMG \
    --enhancer gfpgan \
    --result_dir assets/video/ \
    --checkpoint_dir VisualModel/SadTalker/checkpoints \
    
AUTHOR_VIDEO=$(ls -t assets/video/*.mp4 | head -n 1)
echo "AUTHOR VIDEO = $AUTHOR_VIDEO"

ARTICLE_BASENAME="${ARTICLE##*/}"
ARTICLE_NAME="${ARTICLE_BASENAME%.*}"

ffmpeg \
  -i "$ANCHOR_VIDEO" \
  -i "$AUTHOR_VIDEO" \
  -filter_complex "[0:v]setpts=PTS-STARTPTS[left]; [1:v]setpts=PTS-STARTPTS[right]; [left][right]hstack=inputs=2[v]" \
  -map "[v]" \
  -filter_complex "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[a]" \
  -map "[a]" \
  -c:v libx264 -preset veryfast -crf 18 \
  -c:a aac -b:a 192k \
  assets/video/${ARTICLE_NAME}_podcast.mp4



echo "videos done"
echo


