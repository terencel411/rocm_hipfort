#include <hip/hip_runtime.h>
#include <math.h>

__global__ void gradcal_kernel(double *vel_dev, double *dvel_dev, double dx_dev, int ind)
{
  int ix = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
  int jx = (blockIdx.y - 1) * blockDim.y + threadIdx.y;
  int kx = (blockIdx.z - 1) * blockDim.z + threadIdx.z;

  double vout = 0, num1d60 = 1.0/60.0;
  int im = 16, jm = 16, km = 16;
  int idx1 = 0, idx2 = 0, idx3 = 0;

  if (kx >= 0 && kx < km && jx >= 0 && jx < jm && ix >= 0 && ix < im) {
    idx1 = ix + jx * im + kx * im * jm;

    for (int iter = 1; iter <= 3; iter++) {
      idx2 = idx1 + iter * im * jm * km;
      vout = 0.75 * (vel_dev[idx2] - vel_dev[idx2]) -
             0.15 * (vel_dev[idx2] - vel_dev[idx2]) +
             num1d60 * (vel_dev[idx2] - vel_dev[idx2]);

      idx3 = idx2 + ind * im * jm * km * 3;
      dvel_dev[idx3] = vout / dx_dev;
    }
  }
}

extern "C"
{
  void gradcal_kernel(double **vel_dev, double **dvel_dev, double dx_dev, int ind)
  {
    dim3 blocks(8, 8, 8);
    dim3 grids(8, 8, 8);
    hipError_t err;
    
    hipLaunchKernelGGL(gradcal_kernel, grids, blocks, 0, 0, *vel_dev, *dvel_dev, dx_dev, ind);

    err = hipGetLastError();
    if (err != hipSuccess) {
      printf("HIP ERROR: %s\n", hipGetErrorString(err));
    }

    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
      printf("HIP ERROR after synchronization: %s\n", hipGetErrorString(err));
    }
  }
}
