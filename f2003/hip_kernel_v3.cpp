#include <hip/hip_runtime.h>
#include <math.h>
#include <cstdio>

__global__ void kernel1(double *a_dev, double *b_dev, int n_dev)
{
  int ix = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
  int jx = (blockIdx.y - 1) * blockDim.y + threadIdx.y;
  int idx = 0;

  if (ix>=0 && ix < n_dev && jx>=0 && jx < n_dev) {
    idx = ix + jx * n_dev;
    b_dev[idx] = a_dev[idx] + pow(a_dev[idx], 2);
  }
}

extern "C"
{
  void launch(double **a_dev, double **b_dev, int n_dev)
  {
    dim3 blocks(8, 8, 8);
    dim3 grids(7, 7, 7);
    hipLaunchKernelGGL(kernel1, grids, blocks, 0, 0, *a_dev, *b_dev, n_dev);
  }
}
