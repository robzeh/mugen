import torch
import json
import click
import random
import ast

from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
from tqdm import tqdm
from annoy import AnnoyIndex

from rich.console import Console
console = Console()


# list of len(neighbours), [prompt: neighbour[i] quality]
def prepare_fid_input(desciptions_nn: list):
    pass


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

    # musicgen
    musicgen = MusicGen.get_pretrained("facebook/musicgen-large")
    musicgen.set_generation_params(
        duration=10
    )

    # generate random indices to query
    # random_queries = [random.randint(0, len(metadata)) for _ in range(5)]
    # console.log(f"indices to generate: {random_queries}")

    for i in tqdm(range(0, len(ytids), 8)):
        ytid_b = ytids[i:i+8]  # (b)

        # loop through metadata, get metadata for ytids in batch
        metadata_b = [metadata_item for metadata_item in metadata if metadata_item["ytid"] in ytid_b]  
        metadata_caption_b = [metadata["caption"] for metadata in metadata_b]

        console.log(metadata_caption_b)

        queries_b = [metadata_item["id"] for metadata_item in metadata_b]

        # get nearest neighbors by caption index
        cnn = [caption_index.get_nns_by_item(int(q), 5) for q in queries_b]  # (b, 5)
        console.log(cnn)

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

        all_cfg = []
        all_conditions = []
        for qualities_b in qualities_nn_b:
            attributes, _ = musicgen._prepare_tokens_and_attributes(qualities_b, None)

            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(attributes)
            conditions = attributes + null_conditions
            tokenized = musicgen.lm.condition_provider.tokenize(conditions)

            cfg_conditions = musicgen.lm.condition_provider(tokenized)
            console.log(cfg_conditions["description"][0])
            all_conditions.append(cfg_conditions["description"][0])



            all_cfg.append(cfg_conditions)
        # attributes, _ = musicgen._prepare_tokens_and_attributes(qualities_nn_b, None)

        concat_conditions = torch.cat(all_conditions, dim=1)  # (b, )
        console.log(concat_conditions.shape)

        # null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(attributes)
        # all_conditions = []
        # for cond, null_cond in zip (attributes, null_conditions):
        #     c = cond + null_cond
        #     all_conditions.append(c)

        # console.log(all_conditions)
        # tokenized = musicgen.lm.condition_provider.tokenize(all_conditions)
        # console.log(tokenized)
        # console.log(conditions)

        # tokenized = musicgen.lm.condition_provider.tokenize(conditions)

        # cfg_conditions = musicgen.lm.condition_provider(tokenized)
        # console.log(cfg_conditions)

        # encode each


        # from retrieved items, concat qualities
        # for i in range(len(ytid_b)):

        #     retrieved_qualities_b = []
        #     for nn in cnn:
        #         # (5, len(qualities))
        #         qualities_per_nn = [ast.literal_eval(metadata_item["qualities"]) for metadata_item in metadata if metadata_item["id"] in nn]

        #         PROMPT = f"{metadata_caption_b[i]} This has musical elements of "
        #         flat_sublist = [" ".join(sublist) for sublist in qualities_per_nn]  # (5)
        #         augmented = [f"{PROMPT}{sublist}" for sublist in flat_sublist]  # (5)
        #         console.log(augmented)

        #     # output = musicgen.generate_with_fid(augmented, fid=True)
        #     break
        # break
        break


    # build batch
    # for every neighbour, t5 encode and project
        # attributes, prompt tokens
    # tokens, need this to be max length?



    


if __name__ == "__main__":
    main()