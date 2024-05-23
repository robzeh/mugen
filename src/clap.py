import torch
import json
import laion_clap
import click

from torch.utils.data import DataLoader
from datasets import load_dataset
from annoy import AnnoyIndex
from tqdm import tqdm
from rich.console import Console
console = Console()

from musiccaps_dataset import MusicCapsMetadata

CHECKPOINT_PATH = "/scratch/ssd004/scratch/robzeh/clap_fad/music_audioset_epoch_15_esc_90.14.pt"


# TODO: add youtubeu reference, need to know original audio somehow
# class MusicCapsMetadata:
#     def __init__(self, id, caption, qualities) -> None:
#         self.id = id
#         self.caption = caption
#         self.qualities = qualities

#     def to_dict(self):
#         return {
#             "id": self.id,
#             "caption": self.caption,
#             "qualities": self.qualities
#         }

#     @classmethod
#     def from_dict(cls, d):
#         return cls(
#             id=d["id"],
#             caption=d["caption"],
#             qualities=d["qualities"]
#         )

def get_nn_item_index(index, query, nn):
    return index.get_nns_by_item(query, nn)


def get_nn_vector_index(index, query, nn):
    return index.get_nns_by_vector(query, nn)



@click.command()
@click.option('--build-index', is_flag=True, default=False)
@click.option('--num-trees', default=10)
@click.option('--query', help="index of vector to query", default=0)
@click.option('--nn', default=10)
def main(build_index, num_trees, query, nn):

    # musiccaps
    ds = load_dataset('google/MusicCaps', split='train')
    # ds = ds.select(range(250))

    BATCH_SIZE = 8
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # metadata
    metadata = {}

    # build annoy
    embedding_dim = 512
    caption_index = AnnoyIndex(embedding_dim, 'euclidean')
    qualities_index = AnnoyIndex(embedding_dim, 'euclidean')
    auqualities_index = AnnoyIndex(embedding_dim, 'euclidean')
    aucaption_index = AnnoyIndex(embedding_dim, 'euclidean')

    if build_index:
        console.log(f"Building index with {len(ds)} samples and {num_trees} trees")
        # clap
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        model.load_ckpt(CHECKPOINT_PATH) 

        # embed
        for idx, batch in enumerate(tqdm(dl)):
            # if batch less than 2, skip
            if len(batch["caption"]) < 2:
                continue

            ytid = batch["ytid"]

            captions_b = batch["caption"]
            qualities_b = batch["aspect_list"]

            captions_embed_bd = model.get_text_embedding(captions_b, use_tensor=True)
            qualities_embed_bd = model.get_text_embedding(qualities_b, use_tensor=True)

            # metadata
            for i in range(len(captions_b)):
                metadata[idx * BATCH_SIZE + i] = MusicCapsMetadata(idx * BATCH_SIZE + i, ytid[i], captions_b[i], qualities_b[i])
                caption_index.add_item(idx * BATCH_SIZE + i, captions_embed_bd[i].cpu().detach().numpy())
                qualities_index.add_item(idx * BATCH_SIZE + i, qualities_embed_bd[i].cpu().detach().numpy())
        
        # build the index
        console.log("Building index")
        caption_index.build(10)
        qualities_index.build(10)

        # save index and metadata
        caption_index.save("musiccaps_caption_laionmusic.ann")
        qualities_index.save("musiccaps_qualities_laionmusic.ann")
        with open('musiccaps_metadata_laionmusic.json', 'w') as f:
            f.write(json.dumps([metadata[i].to_dict() for i in metadata], indent=2))
            f.close()

    else:
        caption_index.load('musiccaps_caption_laionmusic.ann')

        aucaption_index.load('musiccaps_caption.ann')
        auqualities_index.load('musiccaps_qualities.ann')

        qualities_index.load('musiccaps_qualities_laionmusic.ann')
        with open('musiccaps_metadata.json', 'r') as f:
            metadata = json.load(f)
            f.close()

        # query
        cnn = get_nn_item_index(caption_index, query, nn)
        aucnn = get_nn_item_index(aucaption_index, query, nn)
        console.log(cnn, aucnn)
        for a, b in zip(cnn, aucnn):
            console.log(metadata[a]["caption"])
            console.log(metadata[b]["caption"])

        auqnn = get_nn_item_index(auqualities_index, query, nn)
        qnn = get_nn_item_index(qualities_index, query, nn)
        console.log(qnn, auqnn)
        for a, b in zip(qnn, auqnn):
            console.log(metadata[a]["qualities"])
            console.log(metadata[b]["qualities"])


@click.command()
def clap():
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(CHECKPOINT_PATH) 

    audio_files = ["/scratch/ssd004/scratch/robzeh/fma_small/000/000140.mp3"]
    audio_embed = model.get_audio_embedding_from_filelist(x = audio_files, use_tensor=True)
    console.log(audio_embed[:,-20:])
    console.log(audio_embed.shape)



if __name__ == "__main__":
    clap()