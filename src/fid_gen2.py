import torch
import json
import click
import ast

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pathlib import Path
from tqdm import tqdm
from annoy import AnnoyIndex

from rich.console import Console
console = Console()

OUTPUT_DIR = "/scratch/ssd004/scratch/robzeh/baselines/fid1"


@click.command()
def main():
    # load index and metadata
    caption_index = AnnoyIndex(512, 'euclidean')
    caption_index.load("musiccaps_caption.ann")
    with open("musiccaps_metadata.json", "r") as f:
        metadata = json.load(f)

    # get list of available ytids
    with open("musiccaps_ytids.txt", "r") as f:
        ytids = f.read().splitlines()

    generated_files = [f.stem for f in Path(OUTPUT_DIR).rglob("*.wav")]
    console.log(generated_files)
    console.log(f"MusicCaps length: {len(ytids)}")
    console.log(f"Already generated {len(generated_files)} files")
    ytids = [ytid for ytid in ytids if ytid not in generated_files]    

    console.log(f"Left to generate: {len(ytids)}")
    if len(ytids) == 0:
        console.log("All files already generated")
        exit(0)

    # musicgen
    musicgen = MusicGen.get_pretrained("facebook/musicgen-large")
    musicgen.set_generation_params(
        duration=10
    )

    for i in tqdm(range(0, len(ytids), 8)):
        ytid_b = ytids[i:i+8]  # (b)

        # loop through metadata, get metadata for ytids in batch
        metadata_b = [metadata_item for metadata_item in metadata if metadata_item["ytid"] in ytid_b]  
        metadata_caption_b = [metadata["caption"] for metadata in metadata_b]

        queries_b = [metadata_item["id"] for metadata_item in metadata_b]

        # get nearest neighbors by caption index
        cnn = [caption_index.get_nns_by_item(int(q), 5) for q in queries_b]  # (b, 5)

        qualities_nn_b = []
        for idx, caption_b in enumerate(cnn):
            query_caption = metadata_caption_b[idx]
            qualities_per_nn = [
                " ".join(ast.literal_eval(metadata_item["qualities"]))
                for metadata_item in metadata if metadata_item["id"] in caption_b
            ]
            caption_qualities_per_nn = [
                " This has musical elements of ".join([query_caption, qualities]) 
                for qualities in qualities_per_nn
            ]
            qualities_nn_b.append(caption_qualities_per_nn)  # (b, 5, len(qualities))

        console.log(qualities_nn_b)

        for idx, qualities in enumerate(qualities_nn_b):
            output = musicgen.generate_with_fid(qualities, fid=True)
            console.log(output.shape)
            for idxx, out in enumerate(output):
                audio_write(f"{OUTPUT_DIR}/{ytid_b[idx]}_{idxx}", out.cpu(), 32_000, strategy="loudness", loudness_compressor=True)
        

if __name__ == "__main__":
    main()
