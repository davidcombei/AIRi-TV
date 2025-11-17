import argparse
import os
import torchaudio
import torch
import random
import re

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def read_wav(path):
    wav, sr = torchaudio.load(path)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav, sr

def silence(duration, sr):
    samples = int(duration * sr)
    return torch.zeros((1, samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", type=str)
    args = parser.parse_args()

    audio_dir = args.audio_dir


    anchor_files = sorted(
        [f for f in os.listdir(audio_dir) if f.endswith("_anchor.wav")],
        key=natural_key
    )
    author_files = sorted(
        [f for f in os.listdir(audio_dir) if f.endswith("_author.wav")],
        key=natural_key
    )

    assert len(anchor_files) == len(author_files), "Anchor and author file count mismatch!"

    anchor_final = []
    author_final = []


    for a_file, u_file in zip(anchor_files, author_files):

        anchor_wav, sr = read_wav(os.path.join(audio_dir, a_file))
        author_wav, _  = read_wav(os.path.join(audio_dir, u_file))

    
        rand_sil = random.uniform(1.0, 2.0)


        anchor_final.append(anchor_wav)
        anchor_final.append(silence(author_wav.shape[-1] / sr, sr))
        anchor_final.append(silence(rand_sil, sr))


        author_final.append(silence(anchor_wav.shape[-1] / sr, sr))
        author_final.append(silence(rand_sil, sr))
        author_final.append(author_wav)



    anchor_out = torch.cat(anchor_final, dim=1)
    author_out = torch.cat(author_final, dim=1)


    torchaudio.save(os.path.join(audio_dir, "anchor_full_audio.wav"), anchor_out, sr)
    torchaudio.save(os.path.join(audio_dir, "author_full_audio.wav"), author_out, sr)

    for f in anchor_files + author_files:
        os.remove(os.path.join(audio_dir, f))

    

if __name__ == "__main__":
    main()
