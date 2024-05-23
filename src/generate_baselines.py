import torch
import json
import click
import ast

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
console = Console()


BATCH_SIZE = 8
OUTPUT_DIR = "/scratch/ssd004/scratch/robzeh/baselines/qcapspost1"

class MusicCapsSample:
    def __init__(self, description):
        self.desc = description

    def to_condition_attributes(self):
        return ConditioningAttributes(
            text={
                "description": self.desc
            },
            wav={"self_wav": WavCondition(
                torch.zeros((1, 1, 1), device="cuda"),
                torch.tensor([0], device="cuda"),
                sample_rate=[32_000],
                path=None
            )}
        )


@click.command()
def main():

    # load metadata
    with open("musiccaps_metadata.json", "r") as f:
        metadata = json.load(f)

    # get list of available ytids
    with open("musiccaps_ytids.txt", "r") as f:
        ytids = f.read().splitlines()

    # get list of already generated ytids, derived from files in output dir
    generated_files = [f.stem for f in Path(OUTPUT_DIR).rglob("*.wav")]
    console.log(generated_files)
    console.log(f"MusicCaps length: {len(ytids)}")
    console.log(f"Already generated {len(generated_files)} files")
    ytids = [ytid for ytid in ytids if ytid not in generated_files]    

    console.log(f"Left to generate: {len(ytids)}")

    # load musicgen
    musicgen = MusicGen.get_pretrained("facebook/musicgen-large")
    musicgen.set_generation_params(
        duration=10
    )

    # for batch size in ytids, get metadata, build captions, generate audio
    for i in tqdm(range(0, len(ytids), BATCH_SIZE)):
        ytid_b = ytids[i:i+BATCH_SIZE]

        metadata_b = [metadata_item for metadata_item in metadata if metadata_item["ytid"] in ytid_b]
        metadata_caption_b = [metadata["caption"] for metadata in metadata_b]
        # console.log(metadata_caption_b)

        metadata_qualities_b = [ast.literal_eval(metadata["qualities"]) for metadata in metadata_b]
        console.log(metadata_qualities_b)

        output_b = musicgen.generate(descriptions=metadata_qualities_b, progress=True)
        console.log(output_b)
        console.log(output_b.shape)

        # save audio
        for idx, one_out in enumerate(output_b):
            audio_write(f"{OUTPUT_DIR}/{ytid_b[idx]}", one_out.cpu(), 32_000, strategy="loudness", loudness_compressor=True)


if __name__ == "__main__":
    main()
