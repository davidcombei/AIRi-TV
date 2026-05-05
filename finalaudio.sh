#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=5

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <anchor_reference.wav> <author_reference.wav> <article.pdf>"
    exit 1
fi

ANCHOR_REF_AUDIO="$1"
AUTHOR_REF_AUDIO="$2"
ARTICLE="$3"

ANCHOR_NAME="${ANCHOR_REF_AUDIO##*/}"
ANCHOR_NAME="${ANCHOR_NAME%.*}"
AUTHOR_NAME="${AUTHOR_REF_AUDIO##*/}"
AUTHOR_NAME="${AUTHOR_NAME%.*}"
ARTICLE_NAME="${ARTICLE##*/}"
ARTICLE_NAME="${ARTICLE_NAME%.*}"

OUTPUT_NAME="${ANCHOR_NAME}_${AUTHOR_NAME}_${ARTICLE_NAME}"

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

ffmpeg \
  -i "assets/audio/anchor_full_audio.wav" \
  -i "assets/audio/author_full_audio.wav" \
  -filter_complex "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[a]" \
  -map "[a]" \
  -c:a aac -b:a 192k \
  "assets/audio/${OUTPUT_NAME}_final.wav"

echo "Audio final generat: assets/audio/${OUTPUT_NAME}_final.wav"
