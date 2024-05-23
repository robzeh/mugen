import os
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import random

from torch.utils.data import DataLoader
from audiocraft.models.loaders import load_compression_model
# from audiocraft.data.audio import audio_write
from datasets import Audio, Dataset
from tqdm import tqdm
from pathlib import Path
from rich.console import Console

console = Console()


info_path = Path("/scratch/ssd004/datasets/nsynth/nsynth-train/examples.json")
with open(info_path, "r") as f:
    data = json.load(f)

train_data_path = Path("/scratch/ssd004/datasets/nsynth/nsynth-train/audio")
train_paths = list(train_data_path.glob("**/*.wav"))
random_subset = random.sample(train_paths, 10_000)
naudio_paths = [str(p) for p in random_subset]

# dataset
nsynth = Dataset.from_dict({
    "audio": naudio_paths
}).cast_column("audio", Audio(sampling_rate=32_000))
console.print(nsynth)

# encodec
encodec = load_compression_model("facebook/musicgen-large", "cuda")
# console.log(encodec.quantizer.vq.layers[0].codebook)
# console.log(encodec.quantizer.vq.layers[0].codebook.shape)

# dataloader
# dl = DataLoader(nsynth, batch_size=1, shuffle=False)

for file in tqdm(nsynth):
    fname, _fext = os.path.splitext(file["audio"]["path"])
    bname = os.path.basename(fname)

    if bname in data:
        instrument_fam = data[bname]["instrument_family_str"]
        instrument_src = data[bname]["instrument_source_str"]
        qualities = data[bname]["qualities_str"]  # list of stirngs
        pitch = data[bname]["pitch"]
        velocity = data[bname]["velocity"]

        x = torch.tensor(file["audio"]["array"])
        x = x.unsqueeze(0).unsqueeze(0)  # (C) => (b,t,c)
        x = x.float().to("cuda")

        with torch.no_grad():
            encoded_frames, emb = encodec.encode(x)

        # console.log(encoded_frames.shape)
        # console.log(emb)
        # console.log(emb.shape)

        # save embedding
        emb = emb[0].tolist()

        # with torch.no_grad():
        #     encoded_frames = encodec.encode(x)
        # tdata = encoded_frames[0].tolist()

        sample = {
            "inputs": emb,
            "instrument_family_str": instrument_fam,
            "instrument_source_str": instrument_src,
            "qualities_str": qualities,
            "pitch": pitch,
            "velocity": velocity,
            "filename": bname
        }

        with open(f"./data/nsynth/train2/{bname}.json", "w") as f:
            json.dump(sample, f)
            f.close()


        # console.print(encoded_frames)

        # # decode encded frames back to audio, torch no grad needed?
        # with torch.no_grad():
        #     decoded_audio = encodec.decode(encoded_frames[0])

        # console.print(decoded_audio.shape)

        # # convert tensor to audio
        # for idx, one_wav in enumerate(decoded_audio):
        #     # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        #     audio_write(f'./data/gen4/{bname}', one_wav.cpu(), encodec.sample_rate, strategy="loudness", loudness_compressor=True)
