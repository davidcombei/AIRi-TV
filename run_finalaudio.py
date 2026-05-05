#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=4

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <anchor_reference.wav> <author_reference.wav> <article.pdf>"
    exit 1
fi

ANCHOR_REF_AUDIO="$1"
AUTHOR_REF_AUDIO="$2"
ARTICLE="$3"

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

echo "Audio final generat in assets/audio/"
echo "  - assets/audio/anchor_full_audio.wav"
echo "  - assets/audio/author_full_audio.wav"
