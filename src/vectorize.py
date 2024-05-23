import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from pathlib import Path
from rich.console import Console
console = Console()

from nsynth_dataset import NsynthDataset

data_files = Path("./data/nsynth/train/")
all_data = list(data_files.glob("**/*.json"))[:5000]
ds = NsynthDataset(all_data)
dl = DataLoader(ds, batch_size=32, shuffle=True)

# embeddings = torch.empty(0, 4*200)

c1, c2, c3, c4 = (
    torch.empty(0, 200),
    torch.empty(0, 200),
    torch.empty(0, 200),
    torch.empty(0, 200),
)

src = torch.empty(0)
fam = torch.empty(0)
# for idx, (inputs, src, fam, note) in enumerate(dl):
for idx, batch in enumerate(dl):
    console.log(batch["inputs"])
    # inputs = batch["inputs"].view(batch["inputs"].shape[0], 4 * 200)
    # embeddings = torch.cat((embeddings, inputs), dim=0)

    # c1 = torch.cat((c1, batch["inputs"][:, 0, :]))
    # c2 = torch.cat((c2, batch["inputs"][:, 1, :]))
    # c3 = torch.cat((c3, batch["inputs"][:, 2, :]))
    # c4 = torch.cat((c4, batch["inputs"][:, 3, :]))
    c1 = torch.cat((c1, batch[:, 0, :]))
    console.log(c1.shape)
    c2 = torch.cat((c2, batch[:, 1, :]))
    c3 = torch.cat((c3, batch[:, 2, :]))
    c4 = torch.cat((c4, batch[:, 3, :]))

    # fam = torch.cat((fam, batch["fam"]), dim=0)
    # src = torch.cat((src, batch["src"]), dim=0)
    # fam = torch.cat((fam, batch), dim=0)
    # src = torch.cat((src, batch), dim=0)

acoustic = [i for i, label in enumerate(src) if label == 0]
electronic = [i for i, label in enumerate(src) if label == 1]
synthetic = [i for i, label in enumerate(src) if label == 2]

# fam
bass = [i for i, label in enumerate(fam) if label == 0]
brass = [i for i, label in enumerate(fam) if label == 1]
flute = [i for i, label in enumerate(fam) if label == 2]
guitar = [i for i, label in enumerate(fam) if label == 3]
keyboard = [i for i, label in enumerate(fam) if label == 4]
mallet = [i for i, label in enumerate(fam) if label == 5]
organ = [i for i, label in enumerate(fam) if label == 6]
reed = [i for i, label in enumerate(fam) if label == 7]
string = [i for i, label in enumerate(fam) if label == 8]
synth_lead = [i for i, label in enumerate(fam) if label == 9]
vocal = [i for i, label in enumerate(fam) if label == 10]

# tsne_model = TSNE(n_components=2, random_state=42, perplexity=35, learning_rate='auto', n_iter=3000)
# tsne_results = tsne_model.fit_transform(embeddings)
# x = tsne_results[:, 0]
# yc1 = tsne_results[:, 1]

# c1_tsne = TSNE(n_components=2, random_state=42, perplexity=35, learning_rate='auto', n_iter=3000)
# c2_tsne = TSNE(n_components=2, random_state=42, perplexity=35, learning_rate='auto', n_iter=3000)
# c3_tsne = TSNE(n_components=2, random_state=42, perplexity=35, learning_rate='auto', n_iter=3000)
# c4_tsne = TSNE(n_components=2, random_state=42, perplexity=35, learning_rate='auto', n_iter=3000)
# c1_results = c1_tsne.fit_transform(c1)
# c2_results = c2_tsne.fit_transform(c2)
# c3_results = c3_tsne.fit_transform(c3)
# c4_results = c4_tsne.fit_transform(c4)
# xc1, yc1 = c1_results[:, 0], c1_results[:, 1]
# xc2, yc2 = c2_results[:, 0], c2_results[:, 1]
# xc3, yc3 = c3_results[:, 0], c3_results[:, 1]
# xc4, yc4 = c4_results[:, 0], c4_results[:, 1]

# umap
# umap_model = umap.UMAP(n_neighbors=20, min_dist=0.1, metric="correlation")
# x_umap = umap_model.fit_transform(embeddings)
# plt.scatter(x_umap[:,0][acoustic], x_umap[:,1][acoustic], label="acoustic")
# plt.scatter(x_umap[:,0][electronic], x_umap[:,1][electronic], label="electronic")
# plt.scatter(x_umap[:,0][synthetic], x_umap[:,1][synthetic], label="synthetic")
# plt.savefig("./umap2.png")

# console.print(tsne_results.shape)

# plt.figure(figsize=(10, 10))
# plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=50)
# plt.savefig("./tsne.png")

# 2d
# plt.figure()
# plt.scatter(x[acoustic], yc1[acoustic], c='r', label='acoustic')
# plt.scatter(x[electronic], yc1[electronic], c='g', label='electronic')
# plt.scatter(x[synthetic], yc1[synthetic], c='b', label='synthetic')
# plt.legend()
# plt.savefig("./src.png")

# plt.figure()
# plt.scatter(xc1[acoustic], yc1[acoustic], label='acoustic')
# plt.scatter(xc1[electronic], yc1[electronic], label='electronic')
# plt.scatter(xc1[synthetic], yc1[synthetic], label='synthetic')
# plt.legend()
# plt.savefig("./figs/c1.png")
# plt.figure()
# plt.scatter(xc2[acoustic], yc2[acoustic], label='acoustic')
# plt.scatter(xc2[electronic], yc2[electronic], label='electronic')
# plt.scatter(xc2[synthetic], yc2[synthetic], label='synthetic')
# plt.legend()
# plt.savefig("./figs/c2.png")
# plt.figure()
# plt.scatter(xc3[acoustic], yc3[acoustic], label='acoustic')
# plt.scatter(xc3[electronic], yc3[electronic], label='electronic')
# plt.scatter(xc3[synthetic], yc3[synthetic], label='synthetic')
# plt.legend()
# plt.savefig("./figs/c3.png")
# plt.figure()
# plt.scatter(xc4[acoustic], yc4[acoustic], label='acoustic')
# plt.scatter(xc4[electronic], yc4[electronic], label='electronic')
# plt.scatter(xc4[synthetic], yc4[synthetic], label='synthetic')
# plt.legend()
# plt.savefig("./figs/c4.png")
# draw single graph for acoustic family and all codebooks
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
# axs[0, 0].scatter(xc1[acoustic], yc1[acoustic], label='acoustic')
# axs[0, 1].scatter(xc2[acoustic], yc2[acoustic], label='acoustic')
# axs[1, 0].scatter(xc3[acoustic], yc3[acoustic], label='acoustic')
# axs[1, 1].scatter(xc4[acoustic], yc4[acoustic], label='acoustic')
# fig.tight_layout()
# fig.legend()
# plt.savefig("./figs/cacoustic.png")

# # draw single graph for electronic family adn all codebooks
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
# axs[0, 0].scatter(xc1[electronic], yc1[electronic], label='electronic')
# axs[0, 1].scatter(xc2[electronic], yc2[electronic], label='electronic')
# axs[1, 0].scatter(xc3[electronic], yc3[electronic], label='electronic')
# axs[1, 1].scatter(xc4[electronic], yc4[electronic], label='electronic')
# fig.tight_layout()
# fig.legend()
# plt.savefig("./figs/celectronic.png")

# # draw single graph for synthetic family and all codebooks
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
# axs[0, 0].scatter(xc1[synthetic], yc1[synthetic], label='synthetic')
# axs[0, 1].scatter(xc2[synthetic], yc2[synthetic], label='synthetic')
# axs[1, 0].scatter(xc3[synthetic], yc3[synthetic], label='synthetic')
# axs[1, 1].scatter(xc4[synthetic], yc4[synthetic], label='synthetic')
# fig.tight_layout()
# fig.legend()
# plt.savefig("./figs/csynthetic.png")




# plt.figure()
# plt.scatter(xc1[bass], yc1[bass], label="bass")
# plt.legend()
# plt.savefig("./figs/c/c1bass.png")
# plt.figure()
# plt.scatter(xc1[brass], yc1[brass], label="brass")
# plt.legend()
# plt.savefig("./figs/c/c1brass.png")
# plt.figure()
# plt.scatter(xc1[flute], yc1[flute], label="flute")
# plt.legend()
# plt.savefig("./figs/c/c1flute.png")
# plt.figure()
# plt.scatter(xc1[guitar], yc1[guitar], label="guitar")
# plt.legend()
# plt.savefig("./figs/c/c1guitar.png")
# plt.figure()
# plt.scatter(xc1[keyboard], yc1[keyboard], label="keyboard")
# plt.legend()
# plt.savefig("./figs/c/c1keyboard.png")
# plt.figure()
# plt.scatter(xc1[mallet], yc1[mallet], label="mallet")
# plt.legend()
# plt.savefig("./figs/c/c1mallet.png")
# plt.figure()
# plt.scatter(xc1[organ], yc1[organ], label="organ")
# plt.legend()
# plt.savefig("./figs/c/c1organ.png")
# plt.figure()
# plt.scatter(xc1[reed], yc1[reed], label="reed")
# plt.legend()
# plt.savefig("./figs/c/c1reed.png")
# plt.figure()
# plt.scatter(xc1[string], yc1[string], label="string")
# plt.legend()
# plt.savefig("./figs/c/c1string.png")
# plt.figure()
# plt.scatter(xc1[synth_lead], yc1[synth_lead], label="synth_lead")
# plt.legend()
# plt.savefig("./figs/c/c1synth.png")
# plt.figure()
# plt.scatter(xc1[vocal], yc1[vocal], label="vocal")
# plt.legend()
# plt.savefig("./figs/c/c1vocal.png")

# plt.figure()
# plt.scatter(xc2[bass], yc2[bass], label="bass")
# plt.legend()
# plt.savefig("./figs/c/c2bass.png")
# plt.figure()
# plt.scatter(xc2[brass], yc2[brass], label="brass")
# plt.legend()
# plt.savefig("./figs/c/c2brass.png")
# plt.figure()
# plt.scatter(xc2[flute], yc2[flute], label="flute")
# plt.legend()
# plt.savefig("./figs/c/c2flute.png")
# plt.figure()
# plt.scatter(xc2[guitar], yc2[guitar], label="guitar")
# plt.legend()
# plt.savefig("./figs/c/c2guitar.png")
# plt.figure()
# plt.scatter(xc2[keyboard], yc2[keyboard], label="keyboard")
# plt.legend()
# plt.savefig("./figs/c/c2keyboard.png")
# plt.figure()
# plt.scatter(xc2[mallet], yc2[mallet], label="mallet")
# plt.legend()
# plt.savefig("./figs/c/c2mallet.png")
# plt.figure()
# plt.scatter(xc2[organ], yc2[organ], label="organ")
# plt.legend()
# plt.savefig("./figs/c/c2organ.png")
# plt.figure()
# plt.scatter(xc2[reed], yc2[reed], label="reed")
# plt.legend()
# plt.savefig("./figs/c/c2reed.png")
# plt.figure()
# plt.scatter(xc2[string], yc2[string], label="string")
# plt.legend()
# plt.savefig("./figs/c/c2string.png")
# plt.figure()
# plt.scatter(xc2[synth_lead], yc2[synth_lead], label="synth_lead")
# plt.legend()
# plt.savefig("./figs/c/c2synth.png")
# plt.figure()
# plt.scatter(xc2[vocal], yc2[vocal], label="vocal")
# plt.legend()
# plt.savefig("./figs/c/c2vocal.png")

# plt.figure()
# plt.scatter(xc3[bass], yc3[bass], label="bass")
# plt.legend()
# plt.savefig("./figs/c/c3bass.png")
# plt.figure()
# plt.scatter(xc3[brass], yc3[brass], label="brass")
# plt.legend()
# plt.savefig("./figs/c/c3brass.png")
# plt.figure()
# plt.scatter(xc3[flute], yc3[flute], label="flute")
# plt.legend()
# plt.savefig("./figs/c/c3flute.png")
# plt.figure()
# plt.scatter(xc3[guitar], yc3[guitar], label="guitar")
# plt.legend()
# plt.savefig("./figs/c/c3guitar.png")
# plt.figure()
# plt.scatter(xc3[keyboard], yc3[keyboard], label="keyboard")
# plt.legend()
# plt.savefig("./figs/c/c3keyboard.png")
# plt.figure()
# plt.scatter(xc3[mallet], yc3[mallet], label="mallet")
# plt.legend()
# plt.savefig("./figs/c/c3mallet.png")
# plt.figure()
# plt.scatter(xc3[organ], yc3[organ], label="organ")
# plt.legend()
# plt.savefig("./figs/c/c3organ.png")
# plt.figure()
# plt.scatter(xc3[reed], yc3[reed], label="reed")
# plt.legend()
# plt.savefig("./figs/c/c3reed.png")
# plt.figure()
# plt.scatter(xc3[string], yc3[string], label="string")
# plt.legend()
# plt.savefig("./figs/c/c3string.png")
# plt.figure()
# plt.scatter(xc3[synth_lead], yc3[synth_lead], label="synth_lead")
# plt.legend()
# plt.savefig("./figs/c/c3synth.png")
# plt.figure()
# plt.scatter(xc3[vocal], yc3[vocal], label="vocal")
# plt.legend()
# plt.savefig("./figs/c/c3vocal.png")

# plt.figure()
# plt.scatter(xc4[bass], yc4[bass], label="bass")
# plt.legend()
# plt.savefig("./figs/c/c4bass.png")
# plt.figure()
# plt.scatter(xc4[brass], yc4[brass], label="brass")
# plt.legend()
# plt.savefig("./figs/c/c4brass.png")
# plt.figure()
# plt.scatter(xc4[flute], yc4[flute], label="flute")
# plt.legend()
# plt.savefig("./figs/c/c4flute.png")
# plt.figure()
# plt.scatter(xc4[guitar], yc4[guitar], label="guitar")
# plt.legend()
# plt.savefig("./figs/c/c4guitar.png")
# plt.figure()
# plt.scatter(xc4[keyboard], yc4[keyboard], label="keyboard")
# plt.legend()
# plt.savefig("./figs/c/c4keyboard.png")
# plt.figure()
# plt.scatter(xc4[mallet], yc4[mallet], label="mallet")
# plt.legend()
# plt.savefig("./figs/c/c4mallet.png")
# plt.figure()
# plt.scatter(xc4[organ], yc4[organ], label="organ")
# plt.legend()
# plt.savefig("./figs/c/c4organ.png")
# plt.figure()
# plt.scatter(xc4[reed], yc4[reed], label="reed")
# plt.legend()
# plt.savefig("./figs/c/c4reed.png")
# plt.figure()
# plt.scatter(xc4[string], yc4[string], label="string")
# plt.legend()
# plt.savefig("./figs/c/c4string.png")
# plt.figure()
# plt.scatter(xc4[synth_lead], yc4[synth_lead], label="synth_lead")
# plt.legend()
# plt.savefig("./figs/c/c4synth.png")
# plt.figure()
# plt.scatter(xc4[vocal], yc4[vocal], label="vocal")
# plt.legend()
# plt.savefig("./figs/c/c4vocal.png")
# # plt.legend()
# # plt.savefig("./figs/c4fam.png")


# # # 3d
# # # fig = plt.figure()
# # # ax = fig.add_subplot(111, projection='3d')
# # # ax.scatter(x[acoustic], yc1[acoustic], z[acoustic], c='r', label='acoustic')
# # # ax.scatter(x[electronic], yc1[electronic], z[electronic], c='g', label='electronic')
# # # ax.scatter(x[synthetic], yc1[synthetic], z[synthetic], c='b', label='synthetic')

# # # plt.legend()
# # # plt.savefig("./scatter21.png")


"""
# idx 0-9
bright, dark, distortion, fast_decay, long_release, multiphonic, nonlinear_env, percussive, reverb, tempo_synced = (
    [], [], [], [], [], [], [], [], [], []
)

    # if qualities indices are in the list, append the token to the list
    # if 0 in qualities_indices:
    #     bright.append(1)
    # else:
    #     bright.append(0)
    # if 1 in qualities_indices:
    #     dark.append(1)
    # else:
    #     dark.append(0)
    # if 2 in qualities_indices:
    #     distortion.append(1)
    # else:
    #     distortion.append(0)
    # if 3 in qualities_indices:
    #     fast_decay.append(1)
    # else:
    #     fast_decay.append(0)
    # if 4 in qualities_indices:
    #     long_release.append(1)
    # else:
    #     long_release.append(0)
    # if 5 in qualities_indices:
    #     multiphonic.append(1)
    # else:
    #     multiphonic.append(0)
    # if 6 in qualities_indices:
    #     nonlinear_env.append(1)
    # else:
    #     nonlinear_env.append(0)
    # if 7 in qualities_indices:
    #     percussive.append(1)
    # else:
    #     percussive.append(0)
    # if 8 in qualities_indices:
    #     reverb.append(1)
    # else:
    #     reverb.append(0)
    # if 9 in qualities_indices:
    #     tempo_synced.append(1)
    # else:
    #     tempo_synced.append(0)

    if 1 in qualities_indices:
        bright.append(idx)
    if 2 in qualities_indices:
        dark.append(idx)
    if 3 in qualities_indices:
        distortion.append(idx)
    if 4 in qualities_indices:
        fast_decay.append(idx)
    if 5 in qualities_indices:
        long_release.append(idx)
    if 6 in qualities_indices:
        nonlinear_env.append(idx)
    if 7 in qualities_indices:
        percussive.append(idx)
    if 8 in qualities_indices:
        reverb.append(idx)
    if 9 in qualities_indices:
        tempo_synced.append(idx)

# c1_tsne = TSNE(n_components=2, random_state=42, perplexity=35, n_iter=3000)
# c1_res = c1_tsne.fit_transform(c1)
# xc1, yc1 = c1_res[:, 0], c1_res[:, 1]

# np.save("./figs/c1/c1_tsne.npy", c1_res)
c1_res = np.load("./figs/c1/c1_tsne.npy")
xc1, yc1 = c1_res[:, 0], c1_res[:, 1]

# plot
# fig, ax = plt.subplots()
# ax.scatter(xc1, yc1, s=1, c="black", alpha=0.5)
# ax.scatter(xc1[bright], yc1[bright], s=1, c="red", alpha=0.5)
# ax.scatter(xc1[dark], yc1[dark], s=1, c="blue", alpha=0.5)

plt.figure()
plt.scatter(xc1[bright], yc1[bright], label="bright")
plt.legend()
plt.savefig("./figs/c1/tsne_bright.png")

plt.figure()
plt.scatter(xc1[dark], yc1[dark], label="dark")
plt.legend()
plt.savefig("./figs/c1/tsne_dark.png")

plt.figure()
plt.scatter(xc1[distortion], yc1[distortion], label="distortion")
plt.legend()
plt.savefig("./figs/c1/tsne_distortion.png")

plt.figure()
plt.scatter(xc1[fast_decay], yc1[fast_decay], label="fast_decay")
plt.legend()
plt.savefig("./figs/c1/tsne_fast_decay.png")

plt.figure()
plt.scatter(xc1[long_release], yc1[long_release], label="long_release")
plt.legend()
plt.savefig("./figs/c1/tsne_long_release.png")

plt.figure()
plt.scatter(xc1[multiphonic], yc1[multiphonic], label="multiphonic")
plt.legend()
plt.savefig("./figs/c1/tsne_multiphonic.png")

plt.figure()
plt.scatter(xc1[nonlinear_env], yc1[nonlinear_env], label="nonlinear_env")
plt.legend()
plt.savefig("./figs/c1/tsne_nonlinear_env.png")

plt.figure()
plt.scatter(xc1[percussive], yc1[percussive], label="percussive")
plt.legend()
plt.savefig("./figs/c1/tsne_percussive.png")

plt.figure()
plt.scatter(xc1[reverb], yc1[reverb], label="reverb")
plt.legend()
plt.savefig("./figs/c1/tsne_reverb.png")

plt.figure()
plt.scatter(xc1[tempo_synced], yc1[tempo_synced], label="tempo_synced")
plt.legend()
plt.savefig("./figs/c1/tsne_tempo_synced.png")

"""
