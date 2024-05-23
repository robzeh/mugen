import torch
import json
import click
import ast

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
console = Console()


BATCH_SIZE = 8
OUTPUT_DIR = "/scratch/ssd004/scratch/robzeh/baselines/ucaps1"


@click.command()
def main():

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

        output_b = musicgen.generate_unconditional(8)

        # save audio
        for idx, one_out in enumerate(output_b):
            audio_write(f"{OUTPUT_DIR}/{ytid_b[idx]}", one_out.cpu(), 32_000, strategy="loudness", loudness_compressor=True)


if __name__ == "__main__":
    main()

