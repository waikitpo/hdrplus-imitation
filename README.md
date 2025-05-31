# Description

This repo implements a simplified **HDR+ multi-frame fusion** and **ISP-like post-processing pipeline** for burst sequences of RAW images (DNG format).  
It includes:
- Multi-scale tile-based optical flow alignment
- Motion-aware merging (dynamic motion mask)
- AWB (joint gray-world + gray-edge)
- CCM
- Global tone mapping (S-curve + sRGB gamma)
- Post-processing: chroma denoise, multi-scale unsharp masking, chromatic aberration correction


# Environments

Developed with Python 3.13

```
pip install -r requirements.txt
```


# How to run

1. Place your test bursts (DNG sequences) into `testing_dataset/`.
2. Modify the `main.py` script if needed (e.g., change the selected folder index).
3. Run the pipeline:

```
python main.py
```


# References

Parts of this pipeline (e.g., burst alignment, HDR+ merging) were inspired by:

## Algorithms and Implementations

- [timothybrooks/hdr-plus](https://github.com/timothybrooks/hdr-plus/)

- [martin-marek/hdr-plus-pytorch](https://github.com/martin-marek/hdr-plus-pytorch/tree/main)

- [amonod/hdrplus-python](https://github.com/amonod/hdrplus-python)

## Articles

- A. Monod, J. Delon and T. Veit, “An Analysis and Implementation of the HDR+ Burst Denoising Method,” Image Processing On Line, vol. 11, pp. 142-169, 2021.

- S. W. Hasinoff, D. Sharlet, R. Geiss, A. Adams, J. T. Barron, F. Kainz, J. Chen and M. Levoy, “Burst Photography for High Dynamic Range and Low-Light Imaging on Mobile Cameras,” ACM Transactions on Graphics, vol. 35, no. 6, pp. 1-12, 2016.

- M. Delbracio, D. Kelly, M. S. Brown and P. Milanfar, “Mobile Computational Photography: A Tour,” Annual Review of Vision Science, 2021.

## Dataset

- All test bursts used for evaluation are from the HDR+ dataset:  [https://hdrplusdata.org/](https://hdrplusdata.org/)
