from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("eda/out")
df = pd.read_csv(OUT / "labels.csv")
meta = pd.read_csv(OUT / "metadata.csv")[["id", "ct_z", "ct_dz"]]
df = df.merge(meta, on="id")
df["num"] = df["id"].str[-2:]
df["contrast"] = df["heart_hu_mean"] > 60
classes = [c for c in ["esophagus", "heart", "trachea", "aorta"] if f"{c}_ml" in df and df[f"{c}_ml"].sum() > 0]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
ax = axes.ravel()

# 1. volume per class per patient
x = np.arange(len(df))
w = 0.8 / len(classes)
for i, c in enumerate(classes):
    ax[0].bar(x + i * w, df[f"{c}_ml"], w, label=c)
ax[0].set_xticks(x + 0.4 - w / 2)
ax[0].set_xticklabels(df["num"])
ax[0].set(ylabel="volume (ml)", title="Label volume per patient (esophagus = esophagus + aorta merged)")
ax[0].legend()

# 2. mean HU inside heart and esophagus mask, contrast split
for c, marker in [("heart", "o"), ("esophagus", "s")]:
    col = np.where(df["contrast"], "tab:orange", "tab:blue")
    ax[1].scatter(x, df[f"{c}_hu_mean"], c=col, marker=marker, label=c)
ax[1].axhline(60, color="k", ls="--", lw=0.8)
ax[1].set_xticks(x)
ax[1].set_xticklabels(df["num"])
ax[1].set(ylabel="mean HU in mask", title="Soft-tissue HU per patient (orange = contrast-enhanced)")
ax[1].legend()

# 3. z extent of each class as horizontal spans, normalised by scan length
for i, r in df.iterrows():
    for j, (c, color) in enumerate([("esophagus", "tab:green"), ("heart", "tab:red"), ("trachea", "tab:purple")]):
        z0, z1 = r[f"{c}_z0"], r[f"{c}_z1"]
        ax[2].plot([z0 / r["ct_z"], z1 / r["ct_z"]], [i + (j - 1) * 0.25] * 2, color=color, lw=3,
                   label=c if i == 0 else None)
ax[2].set_yticks(range(len(df)))
ax[2].set_yticklabels(df["num"])
ax[2].set(xlabel="fraction of scan (z / total slices)", title="Where each organ lives along z")
ax[2].legend(loc="lower right")
ax[2].grid(alpha=0.3)

# 4. fraction of slices with any foreground
df["fg_slices"] = df[["esophagus_z1", "heart_z1", "trachea_z1"]].max(axis=1) - df[["esophagus_z0", "heart_z0", "trachea_z0"]].min(axis=1) + 1
df["fg_frac"] = df["fg_slices"] / df["ct_z"]
ax[3].bar(x, df["fg_frac"], color=np.where(df["ct_dz"] == 2.0, "tab:red", "tab:blue"))
ax[3].set_xticks(x)
ax[3].set_xticklabels(df["num"])
ax[3].set(ylabel="fraction of slices with any label", ylim=(0, 1),
          title="Annotated fraction of each scan (red = dz 2.0 group)")
ax[3].axhline(df["fg_frac"].mean(), color="k", ls="--", lw=0.8)

fig.suptitle("SegTHOR part 1, label analysis")
fig.tight_layout()
fig.savefig(OUT / "labels_overview.png", dpi=110)
print("saved", OUT / "labels_overview.png")
print(df[["id", "contrast", "heart_ml", "esophagus_ml", "fg_slices", "ct_z", "fg_frac"]].round(2).to_string(index=False))
