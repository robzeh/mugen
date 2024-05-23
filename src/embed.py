import torch
import json
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from annoy import AnnoyIndex
# from audiocraft.models.loaders import load_compression_model
from audiocraft.data.audio import audio_write
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
console = Console()

from nsynth_dataset import NsynthDataset


class NsynthMetadata:
    def __init__(self, id, src, fam, qualities, pitch, velocity, fname) -> None:
        self.id = id  # annoy vector index
        self.src = src
        self.fam = fam
        self.qualities = qualities
        self.pitch = pitch
        self.velocity = velocity
        self.fname = fname

    def to_dict(self):
        return {
            "id": self.id,
            "src": self.src,
            "fam": self.fam,
            "qualities": self.qualities,
            "pitch": self.pitch,
            "velocity": self.velocity,
            "fname": self.fname
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d["id"],
            src=d["src"],
            fam=d["fam"],
            qualities=d["qualities"],
            pitch=d["pitch"],
            velocity=d["velocity"],
            fname=d["fname"]
        )
        

data_files = Path("./data/nsynth/train/")
all_data = list(data_files.glob("**/*.json"))[:5000]
ds = NsynthDataset(all_data)
dl = DataLoader(ds, batch_size=8, shuffle=True)

# nsynth metadata dict
metadata = {}

# annoy dimension
dim = 800

# Build the Annoy index
# index = AnnoyIndex(dim, 'euclidean')  # Assuming embeddings are of shape (num_samples, embedding_dim)

token_counts = Counter()

# pitch co occurence
# velocity co occurence
# quality co occurence

for i, batch in enumerate(tqdm(dl)):
    # input_bkt = batch["inputs"]
    # c1 = torch.narrow(input_bkt, 1, 0, 1).squeeze(1)
    # # flatten
    # c1_flat = c1.flatten()
    # token_counts.update(c1_flat.tolist())

    # label
    console.log(batch)

    pitch = batch["pitch"]
    console.log(pitch)

    qualities = batch["qualities"]
    console.log(qualities)

    break

    # count unique tokens
    # unique_tokens, token_counts = torch.unique(c1_flat, return_counts=True)
    # console.log(unique_tokens)
    # console.log(token_counts)


    # # reshape (1,4,200) => (4*200)
    # input_bkt = torch.squeeze(input_bkt)
    # input_bkt = input_bkt.flatten()

    # console.log(input_bkt)

    # # add the item to the index
    # # index.add_item(i, input_bkt)

    # src, fam, pitch, velocity = src.item(), fam.item(), pitch.item(), velocity.item()

    # # create metadata
    # metadata[i] = NsynthMetadata(i, src, fam, qualities, pitch, velocity, fname).to_dict()

# console.log(token_counts.most_common())

# normalize frequences to probabilties
# total_tokens = sum(token_counts.values())
# token_probs = {token: count / total_tokens for token, count in token_counts.items()}
# console.log(token_probs)

# visualize







# # build the index
# index.build(n_trees=10)  

# # Save the index to a file
# index.save("./ns.ann")

# # save the metadata to a file
# with open("./nsynth_metadata.json", "w") as f:
#     json.dump(metadata, f)
#     f.close()

# console.print(index)
# console.print(metadata)

# load annoy index
nsynth_index = AnnoyIndex(dim, 'euclidean')
nsynth_index.load("./ns.ann")

# load metadata
with open("./nsynth_metadata.json", "r") as f:
    metadata = json.load(f)

# nn, distances = nsynth_index.get_nns_by_item(169, 10, include_distances=True)

# console.print(nn)

# for each retrieved vector, print metadata
# for i in nn:
#     console.print(metadata[str(i)])

# for each retrieved vector, reshape back to (4,200)
# retrieved_nn = []
# for i in nn:
#     input_d = nsynth_index.get_item_vector(i)
#     console.print(len(input_d))
#     input_bcd = torch.reshape(torch.tensor(input_d, dtype=torch.long), (4,200))  # (800) => (4,200)
#     retrieved_nn.append(input_bcd)

# num_batches = len(nn) // 5
# stacked, mdata = [], []
# for i in range(num_batches):
#     batch = nn[i*5:(i+1)*5]

#     # reshape
#     reshaped = [torch.tensor(nsynth_index.get_item_vector(i), dtype=torch.long).reshape(4,200) for i in batch]

#     # metadata
#     m = [metadata[str(i)] for i in batch]
#     mdata.extend(m)

#     stacked_tensor = torch.stack(reshaped)
#     stacked.append(stacked_tensor)
#     # input_d = nsynth_index.get_item_vector(i)
#     # console.print(len(input_d))
#     # input_bcd = torch.reshape(torch.tensor(input_d, dtype=torch.long), (4,200))  # (800) => (4,200)
#     # retrieved_nn.append(input_bcd)

# console.print(mdata)

# # load encodec
# encodec = load_compression_model("facebook/musicgen-large", "cuda")

# # decode codes, (B,K,T) => (B,C,T)

# output = []
# # for each retrieved vector, decode and write to audio file, and save as fname from metadata
# for idx, x_kt in enumerate(stacked):
#     with torch.no_grad():
#         x_bct = x_kt.to("cuda")  # (b,k,t)
#         decoded_bct = encodec.decode(x_bct)  
#         console.print(decoded_bct.shape)
#         output.append(decoded_bct)
#     #     audio_write(f'./data/nn/222{idx}', decoded_bct[0].cpu(), encodec.sample_rate, strategy="loudness", loudness_compressor=True)

# # flatten output and write audio
# output = torch.cat(output)
# console.print(output.shape)

# for i, t_out in enumerate(tqdm(output)):
#     fname = mdata[i]["fname"][0]
#     audio_write(f'./data/nn2/169-{i}-{fname}', t_out.cpu(), encodec.sample_rate, strategy="loudness", loudness_compressor=True)


# for each retrieved vector, decode and write to audio file, and save as fname from metadata
# for idx, x_kt in enumerate(retrieved_nn):
#     console.print(x_kt.shape)
#     with torch.no_grad():
#         x_bct = x_kt.unsqueeze(0).to("cuda")  # (b,k,t)
#         decoded_bct = encodec.decode(x_bct)  
#         console.print(decoded_bct.shape)
#         audio_write(f'./data/nn/222{idx}', decoded_bct[0].cpu(), encodec.sample_rate, strategy="loudness", loudness_compressor=True)

