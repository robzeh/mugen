import os
import click
import numpy as np

from frechet_audio_distance import FrechetAudioDistance
from datasets import Audio, Dataset
from scipy.io import wavfile
from scipy.signal import resample
from rich.console import Console
console = Console()


@click.command()
@click.option("--ref-dir")
@click.option("--eval-dir")
def main(ref_dir, eval_dir):

    # print num files in ref and eval
    console.log(f"Reference directory: {ref_dir}")
    console.log(f"Eval directory: {eval_dir}")
    console.log(f"Number of files in reference: {len(os.listdir(ref_dir))}")
    console.log(f"Number of files in evaluation: {len(os.listdir(eval_dir))}")

    # FAD
    frechet = FrechetAudioDistance(
        ckpt_dir="../checkpoints/clap",
        model_name="clap",
        submodel_name="630k-audioset", # for CLAP only
        sample_rate=48_000,
        verbose=True,
        audio_load_worker=8,
        enable_fusion=False, # for CLAP only
    )

    fad_score = frechet.score(ref_dir, eval_dir)
    console.log(f"FAD score: {fad_score}")


if __name__ == "__main__":
    main()