# VolETA++

Code for **VolETA++: A Fully Automatic Depth-Free Radiance-Field Framework for Food 3D Reconstruction and Volume Estimation**, A. AlMughrabi, R. Marques and P. Radeva, CVPR Workshops (3rd MetaFood Workshop), 2026, pp. 9482–9491.

This repository is a snapshot of the VolETA code base (release 0.0.3, March 2025), the version used for VolETA++. The code base was later extended for RASS (NeurIPS 2026); that version is at [github.com/GCVCG/RASS](https://github.com/GCVCG/RASS).

## Contents

- `scripts/`: pipeline stages (COLMAP, FoodMem masks, NeuS2 and Instant-NGP training, scaling, volume estimation).
- `src/`: pose conversion, checkerboard scaling (`src/pose/find_scale.py`), mesh clean-up, scaling and volume computation.
- `docker/`: Dockerfiles for COLMAP, PixSfM, Pre-NeRF 360 and the VolETA environment (see `docker/README.md`).

## Acknowledgement of third-party code

`src/pose/find_scale.py` is adapted from the corner-matched scaling of the ININ-VIAUN team (HR-CMS in the MetaFood CVPR 2024 challenge report), published at [github.com/BITyia/cvpr-metafood](https://github.com/BITyia/cvpr-metafood) (`scale_estimate/find_scale.py`). Our changes add connected-component labelling and convex hulls to isolate the checkerboard before corner detection. The original repository states no licence; that file remains under its authors' terms and is not covered by the licence of this repository.

## Licence

GNU General Public License v3.0 (see `LICENSE`), except where stated above.

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
