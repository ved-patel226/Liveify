import librosa
import numpy as np
import os

sr = 24000
for fname in sorted(os.listdir("../datasetv2/studio")):
    for tag, d in [("studio", "../datasetv2/studio"), ("live", "../datasetv2/live")]:
        path = os.path.join(d, fname)
        if not os.path.exists(path):
            continue
        audio, _ = librosa.load(path, sr=sr, mono=True)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=512)
        zero_cols = (chroma.sum(axis=0) == 0).sum()
        if zero_cols > 0:
            print(
                f"⚠️  {tag:6s} {fname}  {zero_cols}/{chroma.shape[1]} zero-energy frames"
            )
