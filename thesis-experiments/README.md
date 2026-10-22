# Unpublished experiments of the doctoral thesis (October 2026)

Additional results reported in Section 9.8 of the doctoral thesis of Ahmad AlMughrabi, *What You See Is What You Measure? Neural Scene Representations for Accurate Food Volume Estimation* (Universitat de Barcelona, 2026). **They are not published and have not been peer-reviewed.**

They concern the 13 multi-view items of the MetaFood CVPR 2024 challenge used by VolTex and VolETA++:

1. **Variation between training runs.** NeuS2 retrained five times per item (seeds 1337, 1, 2, 3, 4) with the stored VolETA++ inputs and the settings of the paper (15,000 iterations, marching cubes 512, scene scale 0.15). Median coefficient of variation of the volume: 3.0%.
2. **Training settings and masks.** The same inputs with the NeuS2 settings of VolTex (5,000 iterations, 256), and the VolTex inputs (DEVA masks) with those settings, to separate the two factors in which VolTex and VolETA++ differ.
3. **The scale step.** The released checkerboard scale step (`src/pose/find_scale.py`) re-run on the stored dense models.

`python3 stats_repeatability.py` recomputes every number of Section 9.8 from the files here (standard library only; it uses the exact Wilcoxon test of `stats_lineage.py`, the statistics script of the thesis).

## Files

| File | Content |
|---|---|
| `runs_c2_h200.csv` | VolETA++ inputs and settings, NVIDIA H200, 65 runs |
| `runs_c2_3090.csv` | the same on an RTX 3090, 42 runs (same-seed comparison across GPUs) |
| `runs_c2t_mixed.csv` | VolETA++ inputs, VolTex settings, H100 and H200, 65 runs (column `gpu`) |
| `runs_c3_mixed.csv` | VolTex inputs (DEVA masks) and settings, H100 and H200, 65 runs |
| `runs_c1_h100.csv` | an earlier working copy of the inputs (scene scale 0.33), H100, 65 runs |
| `scales.json` | regenerated checkerboard scale per item: metres per COLMAP unit (`ave`), NeuS2 units per COLMAP unit (`k`), threshold (`phi`) |
| `boards_per_threshold.txt` | keyframes with a detected board per binarisation threshold |
| `code/` | the scripts used (see below) |

`volume_raw` is the volume of the closed pieces of the NeuS2 mesh after removing pieces below 5% of the mesh diagonal, in NeuS2 units; `open_faces` counts the faces of open surfaces left out. The `mesh` column records where each mesh was written; the meshes themselves are not included.

## Code

- `code/neus2-seed-headless.patch`: changes to NeuS2 (the copy used by the team, commit 081b819 of github.com/19reborn/NeuS2): a `--seed` option, `--no_eval`, the OpenGL interop include guarded for a build without GUI, and a pybind11 fix for CUDA 12 compilers. NumPy below 2 at run time.
- `code/run_seeds.py`: runs NeuS2 per configuration, item and seed and measures the volume; resumable, `--shard k/n` for parallel workers.
- `code/seeds_mps.sbatch`, `code/seeds_mix.sbatch`: SLURM jobs used on the UB cluster, three runs per GPU through NVIDIA MPS.
- `code/make_bundle.py`, `code/c1_to_c2.py`, `code/colmap_to_nerf.py`: input preparation and unit factors.
- `code/count_boards.py`, `code/evaluate.py`: board detection per threshold and metric volumes with regenerated scales. They use a headless copy of `src/pose/find_scale.py` (display windows and image writes removed, arguments added), not included here because the HR-CMS code it adapts carries no licence.

The input captures are the MetaFood CVPR 2024 challenge data (MetaFood3D) and are not redistributed here.

## Licence

GPL-3.0, as the rest of this repository, except `code/neus2-seed-headless.patch`, which modifies NeuS2 and keeps the NVIDIA Source Code License of NeuS2 (and the BSD licence of pybind11 for its pybind11 lines).
