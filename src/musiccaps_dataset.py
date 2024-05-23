import os
import json
import click
import shutil
import pandas as pd

from pathlib import Path
from rich.console import Console
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
console = Console()


# TODO: add youtubeu reference, need to know original audio somehow
class MusicCapsMetadata:
    def __init__(self, id, ytid, caption, qualities) -> None:
        self.id = id
        self.ytid = ytid
        self.caption = caption
        self.qualities = qualities

    def to_dict(self):
        return {
            "id": self.id,
            "ytid": self.ytid,
            "caption": self.caption,
            "qualities": self.qualities
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d["id"],
            ytid=d["ytid"],
            caption=d["caption"],
            qualities=d["qualities"]
        )

class MusicCapsDataFrame(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index].tolist()


def get_reference_wavs(dir):
    pass
    # for every file in dir, extract ytid, 
    # make a copy of the reference in /scratch/ssd004/scratch/robzeh/MusicCaps and move to dir

# get all ytids in MusicCaps
def get_all_ytids(dir):
    mcaps = Path(dir)
    all_wavs = list(mcaps.glob("*.wav"))
    # split and get list of ytids
    ytids = [wav.stem for wav in all_wavs]
    console.log(ytids)
    console.log(len(ytids))

    # save to file
    with open("musiccaps_ytids.txt", "w") as f:
        for ytid in ytids:
            f.write(f"{ytid}\n")

def print_all_ytids():
    with open("musiccaps_ytids.txt", "r") as f:
        ytids = f.read().splitlines()
    console.log(ytids)
    console.log(len(ytids))


@click.command()
@click.option("--idx", default=0)
@click.option("--input-dir")
def main(idx, input_dir):
    # get_all_ytids(input_dir)
    print_all_ytids()
    # musiccaps
    # ds = load_dataset('google/MusicCaps', split='train')
    # BATCH_SIZE = 8
    # dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    # for idx, batch in enumerate(dl):
    #     console.log(batch)
    #     break

    # df = pd.read_csv("/scratch/ssd004/scratch/robzeh/musiccaps-public-openai.csv")
    # console.log(list(df))
    # ds = MusicCapsDataFrame(df)
    # dl = DataLoader(ds, batch_size=8, shuffle=True)

    # for idx, batch in enumerate(dl):
    #     console.log(batch)
    #     break

    # load metadata
    with open("musiccaps_metadata.json", "r") as f:
        metadata = json.load(f)

    # dir_name = input_dir.split("/")[-1]
    # os.makedirs(f"/scratch/ssd004/scratch/robzeh/musicgen/{dir_name}_ref", exist_ok=True)

    # # for every file in dir, extract ytid,
    # for root, dirs, files in os.walk(input_dir):
    #     for file in files:
    #         if file.endswith(".wav"):
    #             # extract ytid from filename: ytid.wav
    #             ytid = file.split(".")[0]
    #             console.log(ytid)

    #             # make a copy of the reference in /scratch/ssd004/scratch/robzeh/MusicCaps and move to dir
    #             shutil.copy(f"{root}/{file}", f"/scratch/ssd004/scratch/robzeh/musicgen/{dir_name}_ref/{ytid}.wav")
    #             # shutil.move(f"{root}/{file}", f"{input_dir}/{ytid}.wav")

    # make a copy of original wav and mv to 
    # make a copy of the file at location /scratch/ssd004/scratch/robzeh/MusicCaps/{ytid}.wav
    # mv to /scratch/ssd004/scratch/robzeh/musiccaps/{ytid}.wav
    # shutil.copy(f"/scratch/ssd004/scratch/robzeh/MusicCaps/{ytid}.wav", f"/scratch/ssd004/scratch/robzeh/musicgen/b/{ytid}.wav")



if __name__ == "__main__":
    main()