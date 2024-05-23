import os
import torch
import torch.nn.functional as F
import json

from torch.utils.data import Dataset
from rich.console import Console
console = Console()


src_labels = {
    "acoustic": 0,
    "electronic": 1,
    "synthetic": 2
}

fam_labels = {
    "bass": 0,
    "brass": 1,
    "flute": 2,
    "guitar": 3,
    "keyboard": 4,
    "mallet": 5,
    "organ": 6,
    "reed": 7,
    "string": 8,
    "synth_lead": 9,
    "vocal": 10,
}

note_labels = {
    "bright": 0,
    "dark": 1,
    "distortion": 2,
    "fast_decay": 3,
    "long_release": 4,
    "multiphonic": 5,
    "nonlinear_env": 6,
    "percussive": 7,
    "reverb": 8,
    "tempo-synced": 9,
    # "none": 10
}

class NsynthDataset(Dataset):
    def __init__(self, files) -> None:
        super().__init__()
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], "r") as f:
            data = json.load(f)

        filename = os.path.basename(self.files[index])
        
        inputs = torch.tensor(data["inputs"], dtype=torch.float32)
        inputs = torch.squeeze(inputs)  # (1,4,200) => (4,200)

        # src
        instrument_src = data["instrument_source_str"]
        src = src_labels[instrument_src]
        src_tensor = torch.tensor(src, dtype=torch.long)

        # fam
        instrument_fam = data["instrument_family_str"]
        fam = fam_labels[instrument_fam]
        fam_tensor = torch.tensor(fam, dtype=torch.long)

        # note_tensor = torch.empty(0, dtype=torch.long)
        # # console.print(note_quality)
        # if len(note_quality) == 0:
        #     ntensor = torch.tensor([10])
        #     note_tensor = torch.cat((note_tensor, ntensor))
        # else:
        #     for nq in note_quality:
        #         note = note_labels[nq]
        #         # console.print(f"NOTE: {note}")
        #         ntensor = torch.tensor([note], dtype=torch.long)
        #         # console.print(ntensor)
        #         # console.print(ntensor.shape)
        #         # console.print(f"{note_tensor.shape} + {ntensor.shape}")
        #         note_tensor = torch.cat((note_tensor, ntensor), dim=0)

        # note
        # note_quality = data["qualities_str"]

        # max_note_len = len(note_quality)
        # padded = torch.zeros((max_note_len), dtype=torch.long)

        # console.print(max_note_len)
        # console.print(note_quality)

        qualities = data["qualities_str"]
        # console.log(qualities)

        # if len(qualities) == 0:
        #     qualities = ["none"]

        # qtensor = torch.empty((0, len(qualities)))
        # qtensor_intermediate = torch.empty(0)
        q_labels = []
        for q in qualities:
            quality = note_labels[q]
            q_labels.append(quality)

        qlabel_tensor = torch.zeros(1, len(note_labels), dtype=torch.long)
        qlabel_tensor[0, q_labels] = 1
        # console.log(qlabel_tensor)



        # pitch
        pitch = data["pitch"]

        # velocity
        velocity = data["velocity"]
        
        return {
            "filename": filename,
            "inputs": inputs,
            "src": src_tensor, 
            "fam": fam_tensor,
            "qualities": qlabel_tensor,
            "pitch": pitch,
            "velocity": velocity
        }
        # return filename, inputs, src_tensor, fam_tensor, qualities, pitch, velocity
        # return inputs, src_tensor, fam_tensor


class NsynthMetadata:
    def __init__(self, id, src, fam, qualities, pitch, velocity, fname) -> None:
        self.id = id  # annoy vector index
        self.src = src
        self.fam = fam
        self.qualities = qualities
        self.pitch = pitch
        self.velocity = velocity
        self.fname = fname

    def to_dict(self):
        return {
            "id": self.id,
            "src": self.src,
            "fam": self.fam,
            "qualities": self.qualities,
            "pitch": self.pitch,
            "velocity": self.velocity,
            "fname": self.fname
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d["id"],
            src=d["src"],
            fam=d["fam"],
            qualities=d["qualities"],
            pitch=d["pitch"],
            velocity=d["velocity"],
            fname=d["fname"]
        )
