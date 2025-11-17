import argparse
import os
from chatterbox.tts import ChatterboxTTS
import torchaudio as ta
import torch

parser = argparse.ArgumentParser()
parser.add_argument("reference_audio_author", type=str)
parser.add_argument("reference_audio_anchor", type=str)
parser.add_argument("conversation", type=str)
parser.add_argument("output_dir", type=str)
args = parser.parse_args()

ref_author = args.reference_audio_author
ref_anchor = args.reference_audio_anchor
conv_path = args.conversation
outdir = args.output_dir

os.makedirs(outdir, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

tts = ChatterboxTTS.from_pretrained(device=device)

ordered_utterances = []  

with open(conv_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if line.startswith("[ANCHOR]:"):
            text = line.split("]:")[1].strip()
            ordered_utterances.append(("anchor", text))

        elif line.startswith("[AUTHOR]:"):
            text = line.split("]:")[1].strip()
            ordered_utterances.append(("author", text))



anchor_idx = 1
author_idx = 1

for speaker, text in ordered_utterances:

    if speaker == "anchor":
        out_path = f"{outdir}/utterance_{anchor_idx}_anchor.wav"
        wav = tts.generate(
            text=text,
            audio_prompt_path=ref_anchor,
        )
        ta.save(out_path, wav, tts.sr)
        anchor_idx += 1

    else:  
        out_path = f"{outdir}/utterance_{author_idx}_author.wav"
        wav = tts.generate(
            text=text,
            audio_prompt_path=ref_author,
        )
        ta.save(out_path, wav, tts.sr)
        author_idx += 1

print("TTS generation completed successfully.")
