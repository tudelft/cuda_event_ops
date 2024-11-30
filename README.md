# CUDA event warping and splatting

CUDA PyTorch extensions for iterative event warping and splatting.

## Ops

- `iterative_3d_warp`: iterative 3D warping of events following XY flow fields at regular Z intervals to these intervals
- `trilinear_splat`: trilinear splat to images at specified Z values

## Install

```bash
pip install .
# or
pip install git+ssh://git@github.com/Huizerd/cuda_3d_ops
```
