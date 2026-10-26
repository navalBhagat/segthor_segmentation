from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage

SRC = Path("data/segthor_part1/train")
OUT = Path("eda/out")
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}


def load(p):
    return np.asarray(nib.load(p).dataobj)


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    return 2 * inter / (a.sum() + b.sum() + 1e-8)


# ---------- Part 1: Patient_07 GT vs GT2 ----------
p07 = SRC / "Patient_07"
gt1, gt2 = load(p07 / "GT.nii.gz"), load(p07 / "GT2.nii.gz")
ct07 = load(p07 / "Patient_07.nii.gz")
print("=== Patient_07: GT vs GT2")
print(f"identical voxels: {(gt1 == gt2).mean():.6f}  differing voxels: {(gt1 != gt2).sum()}")
print(f"{'class':10s} {'vox GT':>9s} {'vox GT2':>9s} {'dice':>6s} {'HU<-500 GT':>11s} {'HU<-500 GT2':>12s}  centroid GT -> GT2 (voxels)")
for k, name in CLASSES.items():
    m1, m2 = gt1 == k, gt2 == k
    c1 = np.array(ndimage.center_of_mass(m1)) if m1.any() else np.full(3, np.nan)
    c2 = np.array(ndimage.center_of_mass(m2)) if m2.any() else np.full(3, np.nan)
    lung1 = (ct07[m1] < -500).mean() if m1.any() else np.nan
    lung2 = (ct07[m2] < -500).mean() if m2.any() else np.nan
    print(f"{name:10s} {m1.sum():9d} {m2.sum():9d} {dice(m1, m2):6.3f} {lung1:11.3f} {lung2:12.3f}  "
          f"{np.round(c1, 1)} -> {np.round(c2, 1)}  shift={np.round(c2 - c1, 1)}")

# ---------- Part 2: all patients, per-class geometry ----------
rows = []
for pdir in sorted(SRC.glob("Patient_*")):
    pid = pdir.name
    ct = load(pdir / f"{pid}.nii.gz")
    gt = load(pdir / "GT.nii.gz")
    zooms = nib.load(pdir / "GT.nii.gz").header.get_zooms()
    vox_mm3 = float(np.prod(zooms))
    present = sorted(np.unique(gt).tolist())
    row = {"id": pid, "labels": "".join(map(str, present)), "bad_labels": [v for v in present if v > 4]}
    cents = {}
    for k, name in CLASSES.items():
        m = gt == k
        n = int(m.sum())
        row[f"{name}_vox"] = n
        row[f"{name}_ml"] = round(n * vox_mm3 / 1000, 1)
        if n == 0:
            continue
        lab, ncc = ndimage.label(m)
        sizes = np.bincount(lab.ravel())[1:]
        row[f"{name}_ncc"] = int(ncc)
        row[f"{name}_largest_cc_frac"] = round(float(sizes.max() / n), 3)
        row[f"{name}_hu_mean"] = round(float(ct[m].mean()), 1)
        row[f"{name}_lung_frac"] = round(float((ct[m] < -500).mean()), 4)
        c = np.array(ndimage.center_of_mass(m))
        cents[name] = c
        row[f"{name}_cx"], row[f"{name}_cy"], row[f"{name}_cz"] = np.round(c, 1)
        zs = np.where(m.any(axis=(0, 1)))[0]
        row[f"{name}_z0"], row[f"{name}_z1"] = int(zs.min()), int(zs.max())
    if "heart" in cents and "aorta" in cents:
        row["heart_minus_aorta"] = np.round(cents["heart"] - cents["aorta"], 1)
    if "heart" in cents and "trachea" in cents:
        row["heart_minus_trachea"] = np.round(cents["heart"] - cents["trachea"], 1)
    rows.append(row)
    print(f"{pid} done: labels={row['labels']} heart_lung_frac={row.get('heart_lung_frac')} "
          f"heart_ncc={row.get('heart_ncc')} heart_hu={row.get('heart_hu_mean')}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "labels.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

print("\n=== label presence")
print(df[["id", "labels", "bad_labels"]].to_string(index=False))

print("\n=== volume (ml) per class")
print(df[[f"{n}_ml" for n in CLASSES.values()]].describe().T)

print("\n=== connected components and largest-component fraction")
print(df[["id"] + [c for c in df.columns if c.endswith("_ncc") or c.endswith("largest_cc_frac")]].to_string(index=False))

print("\n=== mean HU inside each mask, and fraction of mask below -500 HU")
print(df[["id"] + [c for c in df.columns if c.endswith("_hu_mean") or c.endswith("_lung_frac")]].to_string(index=False))

print("\n=== heart position relative to aorta and trachea (voxel offsets x, y, z)")
print(df[["id", "heart_cz", "aorta_cz", "trachea_cz", "heart_minus_aorta", "heart_minus_trachea"]].to_string(index=False))

print("\n=== z extent per class (first, last slice) and total slices")
print(df[["id"] + [c for c in df.columns if c.endswith("_z0") or c.endswith("_z1")]].to_string(index=False))
