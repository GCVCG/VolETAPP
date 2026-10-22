# Unpublished experiments of the doctoral thesis (October 2026)

Additional results reported in Section 9.8 of the doctoral thesis of Ahmad AlMughrabi, *What You See Is What You Measure? Neural Scene Representations for Accurate Food Volume Estimation* (Universitat de Barcelona, 2026). **They are not published and have not been peer-reviewed.**

They concern the 13 multi-view items of the MetaFood CVPR 2024 challenge used by VolTex and VolETA++:

1. **Variation between training runs.** NeuS2 retrained five times per item (seeds 1337, 1, 2, 3, 4) with the stored VolETA++ inputs and the settings of the paper (15,000 iterations, marching cubes 512, scene scale 0.15). Median coefficient of variation of the volume: 3.0%.
2. **Training settings and masks.** The same inputs with the NeuS2 settings of VolTex (5,000 iterations, 256), and the VolTex inputs (DEVA masks) with those settings, to separate the two factors in which VolTex and VolETA++ differ.
3. **The scale step.** The released checkerboard scale step (`src/pose/find_scale.py`) re-run on the stored dense models.

`python3 stats_repeatability.py`, run in this folder, prints every number of Section 9.8 from the files here, with the identifiers (UR-...) of the thesis's fact ledger (standard library only; it uses the exact Wilcoxon test of `stats_lineage.py`, the statistics script of the thesis, copied unchanged).

## Files

| File | Content |
|---|---|
| `runs_c2_h200.csv` | VolETA++ inputs and settings, NVIDIA H200, 65 runs |
| `runs_c2_3090.csv` | the same on an RTX 3090, 42 runs (same-seed comparison across GPUs) |
| `runs_c2t_mixed.csv` | VolETA++ inputs, VolTex settings, H100 and H200, 65 runs (column `gpu`) |
| `runs_c3_mixed.csv` | VolTex inputs (DEVA masks) and settings, H100 and H200, 65 runs |
| `runs_c1_h100.csv` | an earlier working copy of the inputs (scene scale 0.33), H100, 65 runs |
| `stats_repeatability.py`, `stats_lineage.py` | the statistics (see above) |
| `scales.json` | regenerated checkerboard scale per item: metres per COLMAP unit (`ave`), NeuS2 units per COLMAP unit (`k`), threshold (`phi`) |
| `boards_per_threshold.txt` | keyframes with a detected board per binarisation threshold |
| `code/` | the scripts used (see below) |

The three files written first (`runs_c1_h100.csv`, `runs_c2_h200.csv`, `runs_c2_3090.csv`) have no `gpu` column: their GPU is the one in the file name. `volume_raw` is the volume of the closed pieces of the NeuS2 mesh after removing pieces below 5% of the mesh diagonal, in NeuS2 units; `open_faces` counts the faces of open surfaces left out. The `mesh` column records where each mesh was written; the meshes themselves are not included.

## Code

- `code/neus2-seed-headless.patch`: changes to the copy of NeuS2 used by the team (commit 081b819 of github.com/19reborn/NeuS2, which already carried one local change, `downscale = 1` in `scripts/render_utils.py`, included in the patch): a `--seed` option, `--no_eval` with the pytorch3d import made optional, the OpenGL interop include guarded for a build without GUI, and a pybind11 fix for CUDA 12 compilers. NumPy below 2 at run time.
- `code/run_seeds.py`: runs NeuS2 per configuration, item and seed and measures the volume; resumable, `--shard k/n` for parallel workers.
- `code/seeds_mps.sbatch`, `code/seeds_mix.sbatch`: SLURM jobs used on the UB cluster, three runs per GPU through NVIDIA MPS.
- `code/make_bundle.py`, `code/c1_to_c2.py`, `code/colmap_to_nerf.py`: input preparation and unit factors.
- `code/find_scale-headless.patch`: the headless copy of `src/pose/find_scale.py` used for the scale step (display windows and image writes removed, command-line arguments added). Create it with `cp src/pose/find_scale.py thesis-experiments/code/find_scale_headless.py && patch thesis-experiments/code/find_scale_headless.py thesis-experiments/code/find_scale-headless.patch`.
- `code/count_boards.py`, `code/evaluate.py`: board detection per threshold (needs `find_scale_headless.py`), and metric volumes with the regenerated scales: `python code/evaluate.py runs_c2_h200.csv scales.json`.

The input captures are the MetaFood CVPR 2024 challenge data (MetaFood3D) and are not redistributed here.

## Licence

GPL-3.0, as the rest of this repository, except two patches that keep the terms of the code they modify: `code/neus2-seed-headless.patch` (NVIDIA Source Code License of NeuS2; BSD licence of pybind11 for its pybind11 lines) and `code/find_scale-headless.patch` (the terms of `src/pose/find_scale.py`, adapted from HR-CMS; see the repository README).
