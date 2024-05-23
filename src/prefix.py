import torch
import json
import laion_clap
import click

from torch.utils.data import DataLoader
from datasets import load_dataset
from annoy import AnnoyIndex
from tqdm import tqdm
from datasets import load_dataset
from rich.console import Console
console = Console()

from musiccaps_dataset import MusicCapsMetadata

CHECKPOINT_PATH = "/scratch/ssd004/scratch/robzeh/clap_fad/music_audioset_epoch_15_esc_90.14.pt"

ds = load_dataset("amaai-lab/MusicBench", streaming=True)

model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
model.load_ckpt(CHECKPOINT_PATH) 

text = [
    """
    This mellow instrumental track showcases a dominant electric guitar that opens with a descending riff, followed by arpeggiated chords, hammer-ons, and a slide. 
    The percussion section keeps it simple with rim shots and a common time count, while the bass adds a single note on the first beat of every bar. 
    Minimalist piano chords round out the song while leaving space for the guitar to shine. 
    There are no vocals, making it perfect for a coffee shop or some chill background music. 
    The key is in E major, with a chord progression that centers around that key and a straightforward 4/4 time signature.
    """,
    "The chord progression in this song is E"
]
prefix = model.get_text_embedding(text, use_tensor=True)

console.log(prefix)
console.log(prefix.shape)