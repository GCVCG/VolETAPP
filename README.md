# VolETA++

Code for **VolETA++: A Fully Automatic Depth-Free Radiance-Field Framework for Food 3D Reconstruction and Volume Estimation**, A. AlMughrabi, R. Marques and P. Radeva, CVPR Workshops (3rd MetaFood Workshop), 2026, pp. 9482–9491.

This repository is a snapshot of the VolETA code base (release 0.0.3, March 2025), the version used for VolETA++. The code base was later extended for RASS (NeurIPS 2026); that version is at [github.com/GCVCG/RASS](https://github.com/GCVCG/RASS).

> **Components in the VolETA repository.** Keyframe selection (`key_frame_extraction.sh`, with blur filtering in `blur_fft/` and near-duplicate removal in `imgdp.py`) and featuremetric pose estimation (`pixelsfm.py`) are not in this repository. Use them from [github.com/GCVCG/VolETA-MetaFood](https://github.com/GCVCG/VolETA-MetaFood).

## Contents

- `scripts/`: pipeline stages (COLMAP, FoodMem masks, NeuS2 and Instant-NGP training, scaling, volume estimation).
- `src/`: pose conversion, checkerboard scaling (`src/pose/find_scale.py`), mesh clean-up, scaling and volume computation.
- `docker/`: Dockerfiles for COLMAP, PixSfM, Pre-NeRF 360 and the VolETA environment (see `docker/README.md`).

## Settings used in the paper

- NeuS2: 15,000 iterations, marching-cubes resolution 512, and in each `transforms.json` `"aabb_scale": 1`, `"scale": 0.15`, `"offset": [0.5, 0.5, 0.5]` (`scripts/run_neus2.sh` uses these values).
- Mesh clean-up: isolated pieces below 5% of the mesh diagonal are removed.
- Scale: checkerboard squares of 0.012 m and a binarisation threshold of 240 for the corner detector (`src/pose/find_scale.py`). The threshold is sensitive: on some scenes the board is found in enough keyframes only at 210 to 220.

## Acknowledgement of third-party code

`src/pose/find_scale.py` is adapted from the corner-matched scaling of the ININ-VIAUN team (HR-CMS in the MetaFood CVPR 2024 challenge report), published at [github.com/BITyia/cvpr-metafood](https://github.com/BITyia/cvpr-metafood) (`scale_estimate/find_scale.py`). Our changes add connected-component labelling and convex hulls to isolate the checkerboard before corner detection. The original repository states no licence; that file remains under its authors' terms and is not covered by the licence of this repository.

## Licence

GNU General Public License v3.0 (see `LICENSE`), except for the third-party code below, which remains under the terms of its authors:

- `src/pose/find_scale.py`, adapted from HR-CMS (see above).
- `src/colmap2nerf.py`, from NVIDIA's [instant-ngp](https://github.com/NVlabs/instant-ngp) (`scripts/colmap2nerf.py`), which keeps its NVIDIA copyright notice and the [NVIDIA Source Code License](https://github.com/NVlabs/instant-ngp/blob/master/LICENSE.txt). Helper functions taken from the same script (for example `closest_point_2_lines`) also appear in `src/ARCORE_TO_NERF.py`, `src/ARCORE_TO_NERF_LOGMEAL.py` and `src/pose/pose_extraction.py`.
- `src/colmap/read_write_model.py` and `src/colmap/read_write_dense.py`, from [COLMAP](https://github.com/colmap/colmap) (ETH Zurich and UNC Chapel Hill, BSD licence), which keep their copyright notices.

## Citation

```bibtex
@inproceedings{almughrabi2026voletapp,
  author    = {AlMughrabi, Ahmad and Marques, Ricardo and Radeva, Petia},
  title     = {{VolETA++}: A Fully Automatic Depth-Free Radiance-Field Framework for Food {3D} Reconstruction and Volume Estimation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages     = {9482--9491},
  year      = {2026},
}
```
