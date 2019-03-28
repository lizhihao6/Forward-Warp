
Compile .cu first
```bash
nvcc -c -o src/forward_cuda_kernel.cu.o src/forward_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
```

Then run build.py
```bash
python build.py
```

$$
pos_x = w+flow_x
pos_y = h+flow_y
c_x = ceil(x)
c_y = ceil(y)
f_x = floor(x)
f_y = floor(y)
im1[c_x, c_y] = f_x * f_y * im0[c_x, c_y]
im1[c_x, f_y] = f_x * c_y * im0[c_x, c_y]
im1[f_x, c_y] = c_x * f_y * im0[c_x, c_y]
im1[f_x, f_y] = c_x * c_y * im0[c_x, c_y]
$$