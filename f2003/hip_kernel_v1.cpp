#include <hip/hip_runtime.h>
#include <math.h>

__global__ void gradcal_kernel(double *vel_dev, double *dvel_dev, double dx_dev, int ind)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int jx = blockIdx.y * blockDim.y + threadIdx.y;
  int kx = blockIdx.z * blockDim.z + threadIdx.z;

  double vout = 0, num1d60 = 1.0/60.0;
  int im = 320, jm = 320, km = 320;

  if (kx < km && jx < jm && ix < im) {
    for (int iter = 1; iter <= 3; iter++) {
      int idx1 = (ix+1) + jx*(im+1) + kx*(im+1)*(jm+1) + (iter-1)*(im+1)*(jm+1)*(km+1);
      int idx2 = (ix-1) + jx*(im+1) + kx*(im+1)*(jm+1) + (iter-1)*(im+1)*(jm+1)*(km+1);
      int idx3 = (ix+2) + jx*(im+1) + kx*(im+1)*(jm+1) + (iter-1)*(im+1)*(jm+1)*(km+1);
      int idx4 = (ix-2) + jx*(im+1) + kx*(im+1)*(jm+1) + (iter-1)*(im+1)*(jm+1)*(km+1);
      int idx5 = (ix+3) + jx*(im+1) + kx*(im+1)*(jm+1) + (iter-1)*(im+1)*(jm+1)*(km+1);
      int idx6 = (ix-3) + jx*(im+1) + kx*(im+1)*(jm+1) + (iter-1)*(im+1)*(jm+1)*(km+1);

      vout = 0.75 * (vel_dev[idx1] - vel_dev[idx2]) -
             0.15 * (vel_dev[idx3] - vel_dev[idx4]) +
             num1d60 * (vel_dev[idx5] - vel_dev[idx6]);

      int dvel_idx = ix + jx*(im+1) + kx*(im+1)*(jm+1) + (iter-1)*(im+1)*(jm+1)*(km+1) + (ind-1)*(im+1)*(jm+1)*(km+1)*3;
      
      dvel_dev[dvel_idx] = vout / dx_dev;
    }
  }
}

extern "C"
{
  void gradcal_kernel(double **vel_dev, double **dvel_dev, double dx_dev, int ind)
  {
    dim3 blocks(8, 8, 8);
    dim3 grids((320 + blocks.x - 1) / blocks.x, (320 + blocks.y - 1) / blocks.y, (320 + blocks.z - 1) / blocks.z);
    hipLaunchKernelGGL(gradcal_kernel, grids, blocks, 0, 0, *vel_dev, *dvel_dev, dx_dev, ind);
  }
}
