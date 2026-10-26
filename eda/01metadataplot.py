from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("eda/out")
df = pd.read_csv(OUT / "metadata.csv")

def parse_origin(s):
    return [float(v) for v in s.replace("np.float64(", "").replace(")", "").strip("()").split(",")]

o = np.array([parse_origin(s) for s in df["ct_origin"]])
df["ox"], df["oy"], df["oz"] = o[:, 0], o[:, 1], o[:, 2]
df["fov_mm"] = df["ct_x"] * df["ct_dx"]
df["len_mm"] = df["ct_z"] * df["ct_dz"]
df["group"] = np.where(df["ct_dz"] == 2.0, "A: dz 2.0", "B: dz 2.5")
df["num"] = df["id"].str[-2:]
colors = {"A: dz 2.0": "tab:red", "B: dz 2.5": "tab:blue"}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
ax = axes.ravel()

for g, sub in df.groupby("group"):
    c = colors[g]
    ax[0].scatter(sub["ox"], sub["oy"], c=c, label=g)
    ax[1].scatter(sub["ct_dx"], sub["oy"], c=c, label=g)
    ax[2].scatter(sub["oz"], sub["len_mm"], c=c, label=g)
    ax[3].scatter(sub["ct_dx"], sub["ct_dz"], c=c, label=g)
    ax[4].scatter(sub["ct_z"], sub["len_mm"], c=c, label=g)
    for _, r in sub.iterrows():
        for a, x, y in [(ax[0], r["ox"], r["oy"]), (ax[2], r["oz"], r["len_mm"]), (ax[4], r["ct_z"], r["len_mm"])]:
            a.annotate(r["num"], (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")

xs = np.linspace(200, 360, 10)
ax[0].plot(xs, xs, "k--", lw=0.8, label="x = y (centred)")
ax[0].set(xlabel="origin x (mm)", ylabel="origin y (mm)", title="Origin x vs y")
ax[1].set(xlabel="in-plane spacing dx (mm)", ylabel="origin y (mm)", title="Spacing vs origin y")
ax[2].set(xlabel="origin z (mm)", ylabel="scan length (mm)", title="Table start vs scan length")
ax[3].set(xlabel="dx (mm)", ylabel="dz (mm)", title="Voxel spacing")
ax[4].set(xlabel="slices (z)", ylabel="scan length (mm)", title="Slices vs physical length")

pd.crosstab(df["group"], df["ct_dtype"]).plot.bar(ax=ax[5], rot=0, color=["tab:green", "tab:orange"])
ax[5].set(title="Stored dtype per group", ylabel="patients")

for a in ax[:5]:
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
fig.suptitle("SegTHOR part 1, metadata (20 patients)")
fig.tight_layout()
fig.savefig(OUT / "metadata_overview.png", dpi=110)
print("saved", OUT / "metadata_overview.png")
print(df[["id", "group", "ct_dtype", "ct_dx", "ct_dz", "ct_z", "fov_mm", "len_mm", "ox", "oy", "oz"]]
      .to_string(index=False))
