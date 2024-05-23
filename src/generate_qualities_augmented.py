import torch
import json
import ast
import click
import random

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pathlib import Path
from tqdm import tqdm
from annoy import AnnoyIndex
from rich.console import Console
console = Console()

# OUTPUT_DIR = "/scratch/ssd004/scratch/robzeh/baselines/small_aug10"
QUALITIES_NN_DIR = "/scratch/ssd004/scratch/robzeh/baselines/qualities_nn"
FAR_NN_DIR = "/scratch/ssd004/scratch/robzeh/baselines/far_nn2"
AUG_DIR = "/scratch/ssd004/scratch/robzeh/baselines/small_aug"
# OUTPUT_DIR = "/scratch/ssd004/scratch/robzeh/baselines/postqcaps1"
QUALITY_NNK3 = "/scratch/ssd004/scratch/robzeh/baselines/qualitynnk3"


@click.command()
@click.option("--output_dir", default="", help="Output directory")
@click.option("--caption_index", is_flag=True, default=True, help="Use caption index")
@click.option("--num_k", default=5, help="Number of nearest neighbours to retrieve")
@click.option("--random_neighbours", is_flag=True, default=False, help="Use random neighbours instead of nearest neighbours")
def main(output_dir, caption_index, num_k, random_neighbours):
    
    # load index 
    index = AnnoyIndex(512, "euclidean")
    if caption_index:
        index.load("musiccaps_caption.ann")
    else:
        index.load("musiccaps_qualities.ann")

    # load metadata
    with open("musiccaps_metadata.json", "r") as f:
        metadata = json.load(f)

    # get list of available ytids
    with open("musiccaps_ytids.txt", "r") as f:
        ytids = f.read().splitlines()

    # get list of already generated ytids, derived from files in output dir
    generated_files = [f.stem for f in Path(QUALITY_NNK3).glob("*.wav")]
    console.log(f"MusicCaps length: {len(ytids)}")
    console.log(f"Already generated {len(generated_files)} files")
    ytids = [ytid for ytid in ytids if ytid not in generated_files]
    # ytids = [ytid for ytid in ytids if ytid in generated_files]
    random.shuffle(ytids)
    console.log(f"Left to generate: {len(ytids)}")

    if len(ytids) == 0:
        console.log("All files already generated")
        exit(0)

    # load musicgen
    musicgen = MusicGen.get_pretrained("facebook/musicgen-large")
    musicgen.set_generation_params(
        duration=10
    )

    # for batch size in ytids, get metadata, get retrieved items, build captions and qualities, generate audio
    for i in tqdm(range(0, len(ytids), 8)):
        ytid_b = ytids[i:i+8]

        # metadata_b = [metadata_item for metadata_item in metadata if metadata_item["ytid"] in ytid_b]
        metadata_b = [metadata_item for ytid in ytid_b for metadata_item in metadata if metadata_item["ytid"] == ytid]
        metadata_caption_b = [metadata["caption"] for metadata in metadata_b]

        queries_b = [metadata_item["id"] for metadata_item in metadata_b]

        # get nearest neighbors by caption index
        # cnn = [caption_index.get_nns_by_item(int(q), 5) for q in queries_b]
        qnn = [qualities_index.get_nns_by_item(int(q), 5) for q in queries_b]

        # from retrieved items, concat qualities
        retrieved_qualities_b = []
        for nn in qnn:  # nn is [id, id, id, id, id]
            # random_neighbours = random.sample(nn, 5)
            # [["asdf", "asdf"], ...., 5]
            qualities_per_nn = [ast.literal_eval(metadata_item["qualities"]) for metadata_item in metadata if metadata_item["id"] in nn]
            # string_qualities_per_nn = [f" ".join(qualities) for qualities in qualities_per_nn]
            # flat_list = [set(item for sublist in qualities_per_nn for item in sublist)]
            flat_list = list(set([item for sublist in qualities_per_nn for item in sublist]))

            filtered_qualities = [quality for quality in flat_list if quality.lower() not in ["low quality", "poor audio quality", "amateur recording"]]
            flat_qualities = " ".join(filtered_qualities)
            retrieved_qualities_b.append(flat_qualities)

        # console.log(len(metadata_caption_b))
        # console.log(len(retrieved_qualities_b))
        QUALITIES_PROMPT = "This has elements of "

        # build qualities from retrieved metadata items
        augmented_qualities = [f"{QUALITIES_PROMPT}{qualities}" for qualities in retrieved_qualities_b]
        initial_caption_augmented_qualities = [f"{metadata_caption_b[i]} {augmented_qualities[i]}" for i in range(len(metadata_caption_b))] 
        # initial_caption_augmented_qualities = [f"{metadata_caption_b[i]}" for i in range(len(metadata_caption_b))] 
        console.log(initial_caption_augmented_qualities)

        # generate audio
        output_b = musicgen.generate(descriptions=initial_caption_augmented_qualities, progress=True)

        # save audio
        for idx, one_out in enumerate(output_b):
            audio_write(f"{QUALITY_NNK3}/{ytid_b[idx]}", one_out.cpu(), 32_000, strategy="loudness", loudness_compressor=True)


if __name__ == "__main__":
    main()
