from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd

SRC = Path("data/segthor_part1")
OUT = Path("eda/out")
OUT.mkdir(parents=True, exist_ok=True)


def header_row(nib_obj, prefix: str) -> dict:
    h = nib_obj.header
    x, y, z = nib_obj.shape
    dx, dy, dz = h.get_zooms()
    slope, inter = h.get_slope_inter()
    return {
        f"{prefix}_x": x, f"{prefix}_y": y, f"{prefix}_z": z,
        f"{prefix}_dx": round(float(dx), 4),
        f"{prefix}_dy": round(float(dy), 4),
        f"{prefix}_dz": round(float(dz), 4),
        f"{prefix}_dtype": str(nib_obj.get_data_dtype()),
        f"{prefix}_axcodes": "".join(nib.aff2axcodes(nib_obj.affine)),
        f"{prefix}_qform": int(h["qform_code"]),
        f"{prefix}_sform": int(h["sform_code"]),
        f"{prefix}_slope": slope,
        f"{prefix}_inter": inter,
        f"{prefix}_origin": tuple(np.round(nib_obj.affine[:3, 3], 1)),
    }


rows = []
for pdir in sorted((SRC / "train").glob("Patient_*")):
    pid = pdir.name
    ct = nib.load(pdir / f"{pid}.nii.gz")
    gt = nib.load(pdir / "GT.nii.gz")
    row = {"id": pid, "split": "train"}
    row |= header_row(ct, "ct")
    row |= header_row(gt, "gt")
    row["affine_equal"] = bool(np.allclose(ct.affine, gt.affine))
    row["shape_equal"] = ct.shape == gt.shape
    row["extra_files"] = ",".join(
        f.name for f in sorted(pdir.iterdir())
        if f.name not in (f"{pid}.nii.gz", "GT.nii.gz")
    )
    rows.append(row)

test_dir = SRC / "test"
if test_dir.exists():
    for f in sorted(test_dir.glob("Patient_*.nii.gz")):
        pid = f.name.split(".")[0]
        row = {"id": pid, "split": "test"}
        row |= header_row(nib.load(f), "ct")
        rows.append(row)
else:
    print(f"(no test folder at {test_dir})")

df = pd.DataFrame(rows)
df.to_csv(OUT / "metadata.csv", index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

print("\nCounts")
print(df["split"].value_counts())

cols = ["ct_x", "ct_y", "ct_z", "ct_dx", "ct_dy", "ct_dz"]
print("\n=== shape / spacing summary")
print(df[cols].describe().T)

print("\n=== categorical fields")
for c in ["ct_dtype", "gt_dtype", "ct_axcodes", "gt_axcodes",
          "ct_qform", "ct_sform", "gt_qform", "gt_sform",
          "ct_slope", "ct_inter", "gt_slope", "gt_inter",
          "affine_equal", "shape_equal", "extra_files"]:
    if c in df:
        print(f"{c}: {df[c].value_counts(dropna=False).to_dict()}")

print("\nChecking bounds for violations")
viol = df[
    ~df["ct_dx"].between(0.896, 1.37)
    | (df["ct_dx"] != df["ct_dy"])
    | ~df["ct_dz"].between(2, 3.7)
    | (df["ct_x"] != 512) | (df["ct_y"] != 512)
    | ~df["ct_z"].between(135, 284)
]
print(viol[["id", "split"] + cols].to_string(index=False) if len(viol) else "none")

print("\n=== per patient")
print(df[["id", "split"] + cols + ["ct_axcodes", "affine_equal", "ct_origin", "extra_files"]]
      .to_string(index=False))

print("\nfiles with extra lablels")
for extra in sorted((SRC / "train").glob("Patient_*/GT2.nii.gz")):
    g1 = nib.load(extra.parent / "GT.nii.gz")
    g2 = nib.load(extra)
    print(f"{extra}")
    print(f"  shape   GT={g1.shape}  GT2={g2.shape}")
    print(f"  dtype   GT={g1.get_data_dtype()}  GT2={g2.get_data_dtype()}")
    print(f"  zooms   GT={tuple(round(float(v), 4) for v in g1.header.get_zooms())}"
          f"  GT2={tuple(round(float(v), 4) for v in g2.header.get_zooms())}")
    print(f"  axcodes GT={''.join(nib.aff2axcodes(g1.affine))}  GT2={''.join(nib.aff2axcodes(g2.affine))}")
    print(f"  affine equal: {np.allclose(g1.affine, g2.affine)}")
    if not np.allclose(g1.affine, g2.affine):
        print("  GT affine:\n", np.round(g1.affine, 3))
        print("  GT2 affine:\n", np.round(g2.affine, 3))
