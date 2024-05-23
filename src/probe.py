import os
import json
import torch
import numpy as np

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
console = Console()

from nsynth_dataset import NsynthDataset


info_path = Path("/scratch/ssd004/datasets/nsynth/nsynth-train/examples.json")
with open(info_path, "r") as f:
    data = json.load(f)

data_files = Path("../nsynth/train/")
all_data = list(data_files.glob("**/*.json"))[:500]
ds = NsynthDataset(all_data)
dl = DataLoader(ds, batch_size=32, shuffle=True)

for idx, batch in enumerate(tqdm(dl)):
    console.print(batch)
    break
    # fname, _fext = os.path.splitext(batch[""])
    # bname = os.path.basename(fname)

    # if bname in data:
    #     instrument_fam = data[bname]["instrument_family_str"]
    #     instrument_src = data[bname]["instrument_source_str"]
    #     qualities = data[bname]["qualities_str"]  # list of stirngs

    #     tensor = torch.load(file)

    #     tdata = tensor.tolist()

    #     sample = {
    #         "inputs": tdata,
    #         "instrument_family_str": instrument_fam,
    #         "instrument_source_str": instrument_src,
    #         "qualities_str": qualities
    #     }

    #     # save
    #     with open(f"./data/train_samples/{bname}.json", "w") as f:
    #         json.dump(sample, f)
    #         f.close()
