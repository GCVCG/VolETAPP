"""Copy the inputs of both configurations into bundle/inputs/{c1,c2}/ITEM (transforms.json plus the frames it uses).
Frames listed in a transforms.json but missing on disk are dropped and reported (C1 item 7: 070.png, 071.png)."""
import json, os, shutil

SRC = {"c1": "/media/HDD_4TB_1/amughrabi/data/MTF_Challenge-20240418T205428Z-001/MTF_Challenge",
       "c2": "/media/HDD_16TB/thesis-seeds/c2"}
ITEMS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "13", "14"]
OUT = "/media/HDD_16TB/thesis-seeds/bundle/inputs"

for c, root in (SRC.items() if os.environ.get("BUNDLE") != "c3" else []):
    for i in ITEMS:
        t = json.load(open(f"{root}/{i}/transforms.json"))
        keep = []
        for f in t["frames"]:
            src = os.path.join(root, i, f["file_path"])
            if not os.path.exists(src):
                print(c, i, "missing, dropped:", f["file_path"])
                continue
            dst = os.path.join(OUT, c, i, "rgba", os.path.basename(src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            f["file_path"] = "rgba/" + os.path.basename(src)
            keep.append(f)
        t["frames"] = keep
        json.dump(t, open(f"{OUT}/{c}/{i}/transforms.json", "w"), indent=1)
print("done") if os.environ.get("BUNDLE") != "c3" else None


def bundle_c3():
    """C3, the inputs of VolTex: keyframes of C1 with the DEVA masks of VolTex (masks_deva/Annotations) as alpha,
    NeuS2 scale 0.15 (P3-S9). Frames without an image or a mask are dropped and reported."""
    import numpy as np
    from PIL import Image
    root = SRC["c1"]
    for i in ITEMS:
        t = json.load(open(f"{root}/{i}/transforms.json"))
        t["scale"] = 0.15
        keep = []
        for f in t["frames"]:
            n = os.path.splitext(os.path.basename(f["file_path"]))[0]
            img, msk = f"{root}/{i}/images/{n}.jpg", f"{root}/{i}/masks_deva/Annotations/{n}.png"
            if not (os.path.exists(img) and os.path.exists(msk)):
                print("c3", i, "missing image or mask, dropped:", n)
                continue
            rgb = np.asarray(Image.open(img).convert("RGB"))
            alpha = (np.asarray(Image.open(msk).convert("L")) > 0).astype(np.uint8) * 255
            dst = f"{OUT}/c3/{i}/rgba/{n}.png"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            Image.fromarray(np.dstack([rgb, alpha])).save(dst)
            f["file_path"] = f"rgba/{n}.png"
            keep.append(f)
        t["frames"] = keep
        json.dump(t, open(f"{OUT}/c3/{i}/transforms.json", "w"), indent=1)
        print("c3", i, len(keep), "frames")


if __name__ == "__main__" and os.environ.get("BUNDLE") == "c3":
    bundle_c3()
