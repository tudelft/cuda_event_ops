# CUDA 3D ops

CUDA 3D ops as PyTorch extensions.

## Ops

- `iterative_3d_warp`: iterative 3D warping of events following XY flow fields at regular Z intervals to these intervals
- `trilinear_splat`: trilinear splat to images at specified Z values

## Install

```bash
pip install git+ssh://git@github.com/Huizerd/cuda_3d_ops
```

## To do

- [ ] Check best approach to threads: one per point? Or independent of points?
- [ ] Use accessors to make CUDA code more readable
- [ ] Beautify code generally
