import torch
import json
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from collections import Counter
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
console = Console()

from nsynth_dataset import NsynthDataset, NsynthMetadata


data_files = Path("./data/nsynth/train/")
all_data = list(data_files.glob("**/*.json"))[:5000]
ds = NsynthDataset(all_data)
dl = DataLoader(ds, batch_size=8, shuffle=True)

# co occurrence matrices
src_co_occurrence = Counter()
fam_co_occurrence = Counter()
pitch_co_occurrence = Counter()
velocity_co_occurrence = Counter()
qualities_co_occurrence = Counter()

# c1 = torch.empty(0, 200)
# note_qualitiy = torch.empty(0, 10)

for i, batch in enumerate(tqdm(dl)):
    # extract first codebook
    input_bkt = batch["inputs"]
    c1_bt = torch.narrow(input_bkt, 1, 0, 1).squeeze(1)

    src_b = batch["src"]
    fam_b = batch["fam"]
    pitch_b = batch["pitch"]
    velocity_b = batch["velocity"]
    qualities_bq = batch["qualities"].squeeze(1)  # squeeze?
    note_qualitiy = torch.cat((note_qualitiy, qualities_bq), 0)

    # co occurrence
    for tensor, src, fam, pitch, velocity, qualities in zip(
        c1_bt, src_b, fam_b, pitch_b, velocity_b, qualities_bq
    ):
        # get qualities items
        qualities_idx = qualities.nonzero()
        qualities_indices = qualities_idx[:, 0].tolist()

        for token in tensor:
            t = token.item()
            src_co_occurrence[(t, src.item())] += 1
            fam_co_occurrence[(t, fam.item())] += 1
            pitch_co_occurrence[(t, pitch.item())] += 1
            velocity_co_occurrence[(t, velocity.item())] += 1
            
            # qualities co occurence
            for q in qualities_indices:
                qualities_co_occurrence[(t, q)] += 1

# src_co_occurrence_list = list(src_co_occurrence.items())
# with open("./cooccurrences/src_cooccurrence.json", "w") as f:
#     json.dump(src_co_occurrence_list, f, indent=2)

# fam_co_occurrence_list = list(fam_co_occurrence.items())
# with open("./cooccurrences/fam_cooccurrence.json", "w") as f:
#     json.dump(fam_co_occurrence_list, f, indent=2)

# pitch_co_occurrence_list = list(pitch_co_occurrence.items())
# with open("./cooccurrences/pitch_cooccurrence.json", "w") as f:
#     json.dump(pitch_co_occurrence_list, f, indent=2)

# velocity_co_occurrence_list = list(velocity_co_occurrence.items())
# with open("./cooccurrences/velocity_cooccurrence.json", "w") as f:
#     json.dump(velocity_co_occurrence_list, f, indent=2)

# qualities_co_occurrence_list = list(qualities_co_occurrence.items())
# with open("./cooccurrences/qualities_cooccurrence.json", "w") as f:
#     json.dump(qualities_co_occurrence_list, f, indent=2)

# with open("./cooccurrences/qualities_cooccurrence.json", "r") as f:
#     qualities_co_occurrence_list = json.load(f)

# df = pd.DataFrame(qualities_co_occurrence_list, columns=["token-quality", "count"])
# console.log(df)

# tokens = df["token-quality"].apply(lambda x: x[0]).unique()
# pitches = df["token-quality"].apply(lambda x: x[1]).unique()

# # create emptry df with tokens and labels as rows and columns
# cooccurence_df = pd.DataFrame(index=tokens, columns=pitches)

# # fill df with cooccurrence counts
# for index, row in df.iterrows():
#     # token, pitch, count = row["token-pitch"], row["count"]
#     # cooccurence_df.at[token, pitch] = count
#     cooccurence_df.loc[row["token-quality"][0], row["token-quality"][1]] = row["count"]

# # max_count = cooccurence_df.max().max()

# # # normalize
# # cooccurence_df = cooccurence_df / max_count

# # reset index and convert to numeric type
# # cooccurence_df = cooccurence_df.reset_index().rename(columns={"index": "Token"})
# # cooccurence_df = cooccurence_df.apply(pd.to_numeric, errors="ignore")

# console.log(cooccurence_df)
# console.log(cooccurence_df.sort_values())

# cooccurrence_matrix = cooccurence_df.to_numpy()

# sns.heatmap(cooccurrence_matrix.T, cmap="YlGnBu")
# plt.savefig("./cooccurrences/quality_cooccurrence_heatmap.png")



# pitch_co_occurrence_df = pd.DataFrame(pitch_co_occurrence_list, columns=["token-pitch", "count"])
# pitch_co_occurrence_df[["Token", "Label"]] = pitch_co_occurrence_df["token-pitch"].str.split("-", expand=True)
# pitch_co_occurrence_matrix_df = pitch_co_occurrence_df.pivot(index="Token", columns="Label", values="count")

# sns.heatmap(pitch_co_occurrence_matrix_df, cmap="YlGnBu")

# plt.xlabel("Pitch")
# plt.ylabel("Token")

# plt.savefig("./cooccurrences/pitch_cooccurrence_heatmap.png")
