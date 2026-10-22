"""k12 = C2 NeuS2 units per C1 NeuS2 unit, from the camera centres of the keyframes both workspaces share.

C1 renumbers the keyframes (images/NNN.jpg), so frames are matched by content: 32x24 grey thumbnails,
nearest neighbour, kept when the match is clearly better than the second best.
usage: python c1_to_c2.py ITEM ...   (prints item, k12, relative spread, matched frames)
"""
import json, os, sys
import numpy as np
from PIL import Image

C1 = "/media/HDD_4TB_1/amughrabi/data/MTF_Challenge-20240418T205428Z-001/MTF_Challenge"
C2 = "/media/HDD_4TB_1/amughrabi/data/MTF_Challenge"
C2T = "/media/HDD_16TB/thesis-seeds/c2"


def thumb(p):
    return np.asarray(Image.open(p).convert("L").resize((24, 32)), dtype=np.float32)


def centres(transforms):
    t = json.load(open(transforms))
    return {os.path.splitext(os.path.basename(f["file_path"]))[0]: np.array(f["transform_matrix"])[:3, 3] for f in t["frames"]}


def k12(item):
    c1, c2 = centres(f"{C1}/{item}/transforms.json"), centres(f"{C2T}/{item}/transforms.json")
    t1 = {n: thumb(f"{C1}/{item}/images/{n}.jpg") for n in c1}
    t2 = {n: thumb(f"{C2}/{item}/dense/images/{n}.jpg") for n in c2}
    n2 = list(t2)
    T2 = np.stack([t2[n] for n in n2])
    pairs = []
    for n, t in t1.items():
        d = np.abs(T2 - t).mean(axis=(1, 2))
        a, b = np.argsort(d)[:2]
        if d[a] < 0.5 * d[b]:
            pairs.append((n, n2[a]))
    A = np.array([c1[p] for p, _ in pairs]); B = np.array([c2[q] for _, q in pairs])
    i, j = np.triu_indices(len(pairs), 1)
    r = np.linalg.norm(B[i] - B[j], axis=1) / np.linalg.norm(A[i] - A[j], axis=1)
    return float(np.median(r)), float(np.median(np.abs(r / np.median(r) - 1))), len(pairs)


if __name__ == "__main__":
    for item in sys.argv[1:]:
        print(item, *k12(item), flush=True)
