import os
import torch
import json

# from audiocraft.models import MusicGen
# from audiocraft.models.loaders import load_lm_model
from audiocraft.modules.conditioners import T5Conditioner
from rich.console import Console
console = Console()

# t5
# t5 = load_lm_model("facebook/musicgen-large", "cuda")
t5_conditioner = T5Conditioner("t5-large", output_dim=2048, finetune=False, device="cuda")
console.log(t5_conditioner.device)
t5_conditioner.to("cuda")
console.log(t5_conditioner.device)

# musicgen = MusicGen.get_pretrained("facebook/musicgen-large")

# prepare text conditioning

text = [
    """
    A fusion of 1990s Grunge, Drum and Bass (DnB), House, and Dub elements, characterized by a powerful, distorted breakbeat and a deep, growling bassline reminiscent of Dub and Reggae. Warm, analog synths create evolving chord progressions and soaring melodies, drawing from early House era influences. Sampled dialogue, conversation, and ambient sounds add depth and intrigue. The track ebbs and flows, building to thrilling crescendos and stripping back to its essential components, with a consistent grungy, lo-fi texture and raw, unfiltered energy.
    A fusion of 1990s Grunge, Drum and Bass (DnB), House, and Dub elements, characterized by a powerful, distorted breakbeat and a deep, growling bassline reminiscent of Dub and Reggae. Warm, analog synths create evolving chord progressions and soaring melodies, drawing from early House era influences. Sampled dialogue, conversation, and ambient sounds add depth and intrigue. The track ebbs and flows, building to thrilling crescendos and stripping back to its essential components, with a consistent grungy, lo-fi texture and raw, unfiltered energy.
    A fusion of 1990s Grunge, Drum and Bass (DnB), House, and Dub elements, characterized by a powerful, distorted breakbeat and a deep, growling bassline reminiscent of Dub and Reggae. Warm, analog synths create evolving chord progressions and soaring melodies, drawing from early House era influences. Sampled dialogue, conversation, and ambient sounds add depth and intrigue. The track ebbs and flows, building to thrilling crescendos and stripping back to its essential components, with a consistent grungy, lo-fi texture and raw, unfiltered energy.
    A fusion of 1990s Grunge, Drum and Bass (DnB), House, and Dub elements, characterized by a powerful, distorted breakbeat and a deep, growling bassline reminiscent of Dub and Reggae. Warm, analog synths create evolving chord progressions and soaring melodies, drawing from early House era influences. Sampled dialogue, conversation, and ambient sounds add depth and intrigue. The track ebbs and flows, building to thrilling crescendos and stripping back to its essential components, with a consistent grungy, lo-fi texture and raw, unfiltered energy.
    A fusion of 1990s Grunge, Drum and Bass (DnB), House, and Dub elements, characterized by a powerful, distorted breakbeat and a deep, growling bassline reminiscent of Dub and Reggae. Warm, analog synths create evolving chord progressions and soaring melodies, drawing from early House era influences. Sampled dialogue, conversation, and ambient sounds add depth and intrigue. The track ebbs and flows, building to thrilling crescendos and stripping back to its essential components, with a consistent grungy, lo-fi texture and raw, unfiltered energy.
    """
]

inputs = t5_conditioner.tokenize(text)
i = inputs["input_ids"]
console.log(i.shape)

embeds, mask = t5_conditioner(inputs)
console.log(embeds)
console.log(embeds.shape)
console.log(mask)

# truncate tokenized tokens to 512?