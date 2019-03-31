## Foward Warp Pytorch Version

Has been tested in pytorch=0.4.0, python=3.6, CUDA=9.0

### Install

```bash
export CUDA_HOME=/usr/local/cuda #use your CUDA instead
chmod a+x install.sh
./install.sh
```

### Test

```bash
cd test
python test.py
```

### Usage

```python
from Forward_Warp import forward_warp

fw = forward_warp()
# default interpolation mode is Bilinear
im2_bilinear = fw(im0, flow) 
# use interpolation mode Nearest
# Notice: Nearest input-flow's gradient will be zero when at backward.
im2_nearest = fw(im0, flow, interpolation_mode="Nearest") 
```
