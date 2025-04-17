#include <hip/hip_runtime.h>
#include <math.h>
#include <cstdio>

__global__ void kernel1(int *a_dev, int *b_dev, int n_dev)
{
  int ix = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

  if (ix >= 0 && ix < n_dev) {
    b_dev[ix] = a_dev[ix] + pow(a_dev[ix],2) + pow(a_dev[ix],3) + pow(a_dev[ix],4);
  }
}

extern "C"
{
  void launch(int **a_dev, int **b_dev, int n_dev)
  {
    dim3 blocks1(8, 8, 8);
    dim3 grids1(17, 3, 3);
    hipLaunchKernelGGL(kernel1, grids1, blocks1, 0, 0, *a_dev, *b_dev, n_dev);
  }
}
