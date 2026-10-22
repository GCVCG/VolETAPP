"""Scale k between a transforms.json (NeuS2 units) and the COLMAP model it came from: k = NeuS2 units per COLMAP unit,
the median ratio of camera-centre pair distances over images present in both (matched by file name)."""
import json, os, sys
import numpy as np


def colmap_centres(images_txt):
    out, lines = {}, [l for l in open(images_txt) if not l.startswith("#")]
    for l in lines[::2]:  # each image takes two lines
        p = l.split()
        q, t = np.array(p[1:5], float), np.array(p[5:8], float)
        w, x, y, z = q
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
        out[os.path.splitext(p[9])[0]] = -R.T @ t
    return out


def k_factor(transforms, images_txt):
    t = json.load(open(transforms))
    a = {os.path.splitext(os.path.basename(f["file_path"]))[0]: np.array(f["transform_matrix"])[:3, 3] for f in t["frames"]}
    b = colmap_centres(images_txt)
    names = sorted(set(a) & set(b))
    A, B = np.array([a[n] for n in names]), np.array([b[n] for n in names])
    i, j = np.triu_indices(len(names), 1)
    r = np.linalg.norm(A[i] - A[j], axis=1) / np.linalg.norm(B[i] - B[j], axis=1)
    return float(np.median(r)), float(r.std() / np.median(r)), len(names)


if __name__ == "__main__":
    print(k_factor(sys.argv[1], sys.argv[2]))
