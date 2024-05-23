import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
console = Console()

from nsynth_dataset import NsynthDataset


class ProbingClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


data_files = Path("./data/train_samples/")
all_data = list(data_files.glob("**/*.json"))
ds = NsynthDataset(all_data)

# dl
dl = DataLoader(ds, batch_size=8, shuffle=True)

# model and related
c1_model, c2_model, c3_model, c4_model = (
    ProbingClassifier(200, 64, 3),
    ProbingClassifier(200, 64, 3),
    ProbingClassifier(200, 64, 3),
    ProbingClassifier(200, 64, 3)
)
c1_model, c2_model, c3_model, c4_model = (
    c1_model.cuda(),
    c2_model.cuda(),
    c3_model.cuda(),
    c4_model.cuda()
)

criterion = nn.CrossEntropyLoss()

LR = 0.001
c1_optim, c2_optim, c3_optim, c4_optim = (
    optim.Adam(c1_model.parameters(), lr=LR),
    optim.Adam(c2_model.parameters(), lr=LR),
    optim.Adam(c3_model.parameters(), lr=LR),
    optim.Adam(c4_model.parameters(), lr=LR)
)

for epoch in tqdm(range(10)):

    for idx, (inputs, label) in enumerate(tqdm(dl)):
        c1_optim.zero_grad()
        c2_optim.zero_grad()
        c3_optim.zero_grad()
        c4_optim.zero_grad()

        c1 = torch.narrow(inputs, 1, 0, 1).squeeze(1)
        c2 = torch.narrow(inputs, 1, 1, 1).squeeze(1)
        c3 = torch.narrow(inputs, 1, 2, 1).squeeze(1)
        c4 = torch.narrow(inputs, 1, 3, 1).squeeze(1)

        # targets
        targets = F.one_hot(label, num_classes=3)
        targets = torch.tensor(targets, dtype=torch.float32)
        targets = targets.to("cuda")

        c1 = c1.to("cuda")
        out1 = c1_model(c1)
        loss1 = criterion(out1, targets)
        loss1.backward()
        c1_optim.step()

        c2 = c2.to("cuda")
        out2 = c2_model(c2)
        loss2 = criterion(out2, targets)
        loss2.backward()
        c2_optim.step()

        c3 = c3.to("cuda")
        out3 = c3_model(c3)
        loss3 = criterion(out3, targets)
        loss3.backward()
        c3_optim.step()

        c4 = c4.to("cuda")
        out4 = c4_model(c4)
        loss4 = criterion(out4, targets)
        loss4.backward()
        c4_optim.step()


        if idx % 100 == 0:
            console.print(
                f"loss1: {loss1} \nloss2: {loss2} \nloss3: {loss3} \nloss4: {loss4}"
            )
        
        # console.print(loss)

# torch.save({
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     "loss": loss
# }, "./chekcpoint1.pth")