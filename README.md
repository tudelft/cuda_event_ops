# CUDA event warping and splatting

CUDA PyTorch extensions for iterative warping and splatting of events.

## Ops

- `iterative_3d_warp`: iterative 3D warping of events following XY flow fields at regular Z intervals to these intervals
- `trilinear_splat`: trilinear splat to images at specified Z values

## Install

```bash
pip install .
# or
pip install git+ssh://git@github.com/tudelft/cuda_event_ops
```
