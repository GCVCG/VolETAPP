"""Run NeuS2 per configuration, item and seed on the 13 multi-view MetaFood items, and measure the volume.

Configurations (inputs and NeuS2 settings):
  c1   an October 2024 working copy of the inputs (keyframes; all 199 frames for item 1), scene scale 0.33,
       15,000 iterations, marching cubes 512
  c2   the stored VolETA++ inputs: keyframe poses with the food images, scene scale 0.15, 15,000 at 512
  c2t  the inputs of c2 with the settings of VolTex: 5,000 iterations, marching cubes 256
  c3   the inputs of VolTex: the keyframes of c1 with the DEVA masks of VolTex (make_bundle.py), 5,000 at 256
The volume is that of the closed pieces after removing pieces below 5% of the mesh diagonal, in NeuS2 units.

usage: python run_seeds.py --config c2 [--items 1 2 ...] [--seeds 1337 1 2 3 4] [--shard k/n] [--dry]
       python run_seeds.py --config c2 --remeasure      (recompute the volumes of the meshes already run)
Results: results.csv (c1) or results_<config>.csv; resumable: an (item, seed) pair already there is skipped.
"""
import argparse, csv, fcntl, os, shutil, subprocess, sys, time

ROOT = os.environ.get("SEEDS_ROOT", "/media/HDD_16TB/thesis-seeds")
DATA = {"c1": "/media/HDD_4TB_1/amughrabi/data/MTF_Challenge-20240418T205428Z-001/MTF_Challenge",
        "c2": f"{ROOT}/c2"}  # transforms_ref.json keyframe poses with the food images, NeuS2 scale 0.15
DATA.update({"c2t": DATA["c2"], "c3": f"{ROOT}/inputs/c3"})
STEPS = {"c1": 15000, "c2": 15000, "c2t": 5000, "c3": 5000}
RES = {"c1": 512, "c2": 512, "c2t": 256, "c3": 256}
if "SEEDS_ROOT" in os.environ:  # cluster copy: inputs under ROOT/inputs/<config> (make_bundle.py); c2t reads those of c2
    DATA = {c: f"{ROOT}/inputs/{'c2' if c == 'c2t' else c}" for c in DATA}
NEUS2 = f"{ROOT}/NeuS2-seed"
PY_NEUS2 = os.environ.get("PY_NEUS2", os.path.expanduser("~/anaconda3/envs/neus2_voletar/bin/python"))
PY_MESH = os.environ.get("PY_MESH", os.path.expanduser("~/anaconda3/envs/voletar/bin/python"))
ITEMS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "13", "14"]
SEEDS = [1337, 1, 2, 3, 4]
OUT = f"{ROOT}/runs"
FIELDS = ["item", "seed", "volume_raw", "open_faces", "train_seconds", "mesh", "gpu"]

VOLUME = r"""
import sys, numpy as np, trimesh
m = trimesh.load(sys.argv[1], force="mesh")
d = np.linalg.norm(m.extents)
# clean-up of VolETA: drop pieces whose diagonal is below 5% of the mesh diagonal
parts = [p for p in m.split(only_watertight=False) if np.linalg.norm(p.extents) >= 0.05 * d]
# volume of the closed pieces only: NeuS2 sometimes adds a large open sheet away from the food, which has no volume
print(sum(abs(p.volume) for p in parts if p.is_watertight), sum(len(p.faces) for p in parts if not p.is_watertight))
"""


def csv_path(config):
    return f"{ROOT}/results.csv" if config == "c1" else f"{ROOT}/results_{config}.csv"


def done(config):
    if not os.path.exists(csv_path(config)):
        return set()
    return {(r["item"], r["seed"]) for r in csv.DictReader(open(csv_path(config)))}


def run(item, seed, dry, config):
    name = f"mtf{item}_s{seed}" if config == "c1" else f"{config}_mtf{item}_s{seed}"
    mesh = f"{OUT}/{name}.obj"
    cmd = [PY_NEUS2, "-u", "scripts/run.py", "--scene", f"{DATA[config]}/{item}/transforms.json", "--name", name,
           "--network", "dtu.json", "--n_steps", str(STEPS[config]), "--seed", str(seed),
           "--save_mesh", "--marching_cubes_res", str(RES[config]), "--no_eval"]
    print(" ".join(cmd), flush=True)
    if dry:
        return None
    t = time.time()
    with open(f"{OUT}/{name}.log", "w") as log:
        subprocess.run(cmd, cwd=NEUS2, stdout=log, stderr=subprocess.STDOUT, check=True)
    secs = round(time.time() - t)
    shutil.move(f"{NEUS2}/output/{name}/mesh/{STEPS[config]}.obj", mesh)  # run.py ignores --save_mesh_path
    shutil.rmtree(f"{NEUS2}/output/{name}", ignore_errors=True)  # checkpoints are not needed
    vol, open_faces = measure(mesh)
    return dict(item=item, seed=seed, volume_raw=vol, open_faces=open_faces, train_seconds=secs, mesh=mesh, gpu=os.environ.get("GPU_NAME", ""))


def measure(mesh):
    return subprocess.run([PY_MESH, "-c", VOLUME, mesh], capture_output=True, text=True, check=True).stdout.split()


def remeasure(config):
    """Recompute the volume of every mesh already in the results file of a configuration."""
    rows = list(csv.DictReader(open(csv_path(config))))
    for r in rows:
        r["volume_raw"], r["open_faces"] = measure(r["mesh"])
        r.pop("faces", None)
    with open(csv_path(config), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--items", nargs="*", default=ITEMS)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--config", default="c1", choices=sorted(DATA))
    p.add_argument("--dry", action="store_true")
    p.add_argument("--remeasure", action="store_true")
    p.add_argument("--shard", default="0/1", help="k/n: run only every n-th (seed, item) pair, starting at k (parallel workers)")
    a = p.parse_args()
    if a.remeasure:
        return remeasure(a.config)
    os.makedirs(OUT, exist_ok=True)
    finished = done(a.config)
    new = not os.path.exists(csv_path(a.config))
    with open(csv_path(a.config), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new and not a.dry:
            w.writeheader(); f.flush()  # before any row of a parallel worker
        k, n = map(int, a.shard.split("/"))
        pairs = [(seed, item) for seed in a.seeds for item in a.items]  # seed-major: every item gets a first run early
        for seed, item in pairs[k::n]:
            if (item, str(seed)) in finished:
                continue
            row = run(item, seed, a.dry, a.config)
            if row:
                fcntl.flock(f, fcntl.LOCK_EX)  # workers share the results file
                w.writerow(row); f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
                print(row, flush=True)


if __name__ == "__main__":
    sys.exit(main())
