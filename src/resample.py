import os
import click
import torchaudio
import shutil

from scipy.io import wavfile
from scipy.signal import resample
from rich.console import Console
console = Console()


@click.command()
@click.option("--input")
@click.option("--out-dir")
@click.option("--ref-dir")
@click.option("--eval-dir")
@click.option("--sr", help="get sample rate of a file")
def main(input, out_dir, ref_dir, eval_dir, sr):
    # get sample rate of a file
    if sr:
        sr, d = wavfile.read(os.path.abspath(sr))
        console.log(sr)

    # for every wavfile in input, resample to 48k and save to out_dir
    if input:
        for root, dirs, files in os.walk(input):
            console.log(f"Resampling {len(files)} files in {root}")
            for file in files:
                if file.endswith(".wav"):
                    # read
                    # sr, d = wavfile.read(os.path.join(root, file))
                    audio, sr = torchaudio.load(os.path.join(root, file))

                    console.log(audio)
                    console.log(sr)

                    # resample
                    # s = resample(d, 48_000)

                    # # save
                    # wavfile.write(os.path.join(out_dir, file), 48_000, s)


if __name__ == "__main__":
    main()