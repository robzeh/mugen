import torch
import json
import click
import ast
import random

from audiocraft.models import MusicGen
# from audiocraft.models.loaders import load_compression_model
from audiocraft.data.audio import audio_write
from tqdm import tqdm
from annoy import AnnoyIndex

from rich.console import Console
console = Console()


@click.command()
@click.option('--query', help="index of vector to query", default=0)
@click.option('--nn', default=5)
@click.option('--num-samples', default=1)
@click.option('--dur', help="duration of generated audio", default=10)
@click.option('--output', help="output directory", default="/scratch/ssd004/scratch/robzeh/musicgen/d")
def main(query, nn, num_samples, dur, output):
    # load index and metadata
    caption_index = AnnoyIndex(512, 'euclidean')
    caption_index.load("musiccaps_caption.ann")
    qualities_index = AnnoyIndex(512, 'euclidean')
    qualities_index.load("musiccaps_qualities.ann")
    with open("musiccaps_metadata.json", "r") as f:
        metadata = json.load(f)

    # load musicgen
    musicgen = MusicGen.get_pretrained("facebook/musicgen-large")
    musicgen.set_generation_params(
        duration=dur
    )

    # generate random indices to query
    random_queries = [random.randint(0, len(metadata)) for _ in range(num_samples)]
    console.log(f"indices to generate: {random_queries}")

    # for each random index, get nearest neighbors, build captions and qualities, and generate audio
    for q in tqdm(random_queries):
        # get nearest neighbors
        cnn = caption_index.get_nns_by_item(int(q), nn)
        qnn = qualities_index.get_nns_by_item(int(q), nn)

        # build captions list
        captions = " ".join([metadata[idx]["caption"] for idx in cnn])
        captions = [captions]
        # console.log(captions)

        # build qualities list
        # qualities is a literal string that is a list of qualities
        qualities_list = []
        console.log(qnn)
        for idx in qnn:
            qualities_list.append(ast.literal_eval(metadata[idx]["qualities"]))
        qualities_flat = [quality for qualities in qualities_list for quality in qualities]
        qualities = " ".join(qualities_flat)
        qualities = [qualities]

        output_bt, tokens = musicgen.generate(descriptions=qualities, progress=True, return_tokens=True)

        # console.log(output_bt)
        # console.log(output_bt.shape)

        # get metadata ytid
        ytid = metadata[q]["ytid"]

        for idx, one_out in enumerate(output_bt):
            # resample here?
            audio_write(f"{output}/{ytid}", one_out.cpu(), 48_000, strategy="loudness", loudness_compressor=True)


    # get nearest neighbors
    cnn = caption_index.get_nns_by_item(int(query), nn)
    qnn = qualities_index.get_nns_by_item(int(query), nn)

    # # build captions list
    # captions = " ".join([metadata[idx]["caption"] for idx in cnn])
    # captions = [captions]
    # # console.log(captions)

    # # build qualities list
    # # qualities is a literal string that is a list of qualities
    # qualities_list = []
    # console.log(qnn)
    # for idx in qnn:
    #     qualities_list.append(ast.literal_eval(metadata[idx]["qualities"]))
    # qualities_flat = [quality for qualities in qualities_list for quality in qualities]
    # qualities = " ".join(qualities_flat)
    # qualities = [qualities]

    # output_bt, tokens = musicgen.generate(descriptions=qualities, progress=True, return_tokens=True)

    # console.log(output_bt)
    # console.log(output_bt.shape)

    # for idx, one_out in enumerate(output_bt):
    #     audio_write(f"{output}/{idx}", one_out.cpu(), musicgen.sample_rate, strategy="loudness", loudness_compressor=True)



"""
# prompt_tokens = []
# # for each retrieved vector, print metadata
# for i in nn:
#     console.print(metadata[str(i)])

#     input_d = nsynth_index.get_item_vector(i)  # (800)
#     input_kt = torch.reshape(torch.tensor(input_d, dtype=torch.long), (4, 200))  
#     prompt_tokens.append(input_kt)


# # musicgen
# model = MusicGen.get_pretrained("facebook/musicgen-large")
# model.set_generation_params(
#     duration=10
# )

# attributes, _ = model._prepare_tokens_and_attributes([None] * 10, prompt=None)
# console.log(attributes)

# # batch to 10
# prompt_tokens_bkt = torch.stack(prompt_tokens)
# console.log(prompt_tokens_bkt)

# # p = prompt_tokens[0].unsqueeze(0)
# # console.log(p.shape)
# # generate
# gen_tokens_bct = model._generate_tokens(attributes, prompt_tokens_bkt, True)
# console.log(gen_tokens_bct.shape)
# # console.log(gen_tokens_bct)

# # decode to audio
# gen_audio_bct = model.generate_audio(gen_tokens_bct)
# console.log(gen_audio_bct.shape)

# # write to audio
# for idx, one_wav in enumerate(gen_audio_bct):
#     # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
#     audio_write(f'./data/gen5/{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

"""


if __name__ == "__main__":
    main()