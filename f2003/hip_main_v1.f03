module vars
  use iso_c_binding
  use hipfort
  use hipfort_check
  implicit none

  integer(c_int), parameter :: im=320, jm=320, km=320, hm=5
  integer(c_int), parameter :: im_dev=320, jm_dev=320, km_dev=320, hm_dev=5

  real(8), parameter :: pi=4.d0*atan(1.0_8), num1d60=1.d0/60.d0

  type(dim3) :: grids, blocks
  integer :: istat
  character(len=100) :: error_string
  real :: tstart, tfinish, ctime(12)
  
  real(8) :: dx, dy, dz
  real(8), allocatable, dimension(:,:,:,:) :: x, vel
  real(8), allocatable, dimension(:,:,:,:,:) :: dvel

  type(c_ptr) :: vel_dev = c_null_ptr
  type(c_ptr) :: dvel_dev = c_null_ptr
  type(c_ptr) :: dx_dev, dy_dev, dz_dev
  integer(c_int) :: ind
end module vars

module kernels
  use iso_c_binding
  use hipfort
  use hipfort_check
  use vars
  implicit none

  interface
     subroutine gradcal_kernel(vel_dev, dvel_dev, dx_dev, ind) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr) :: vel_dev, dvel_dev
       type(c_ptr) :: dx_dev, dy_dev, dz_dev
       integer(c_int) :: ind
     end subroutine
  end interface
end module kernels

program gpu_comp
  use iso_c_binding
  use hipfort
  use hipfort_check
  use vars
  use kernels
  implicit none

  integer :: i, j, k

  interface
     function hipMalloc(ptr, size) bind(c, name="hipMalloc")
       use iso_c_binding
       implicit none
       type(c_ptr) :: ptr
       integer(c_size_t), value :: size
       integer(c_int) :: hipMalloc
     end function hipMalloc
  end interface

  allocate(x(0:im,0:jm,0:km,1:3))
  allocate(vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:3))
  allocate(dvel(0:im,0:jm,0:km,1:3,1:3))

  dx = 2.d0 * pi / dble(im)
  dy = 2.d0 * pi / dble(jm)
  dz = 2.d0 * pi / dble(km)

  do k = 0, km
    do j = 0, jm
      do i = 0, im
        x(i,j,k,1) = 2.d0 * pi / dble(im) * dble(i)
        x(i,j,k,2) = 2.d0 * pi / dble(jm) * dble(j)
        x(i,j,k,3) = 2.d0 * pi / dble(km) * dble(k)
        vel(i,j,k,1) = sin(x(i,j,k,1)) * cos(x(i,j,k,2)) * cos(x(i,j,k,3))
        vel(i,j,k,2) = -cos(x(i,j,k,1)) * sin(x(i,j,k,2)) * cos(x(i,j,k,3))
        vel(i,j,k,3) = 0.d0
      end do
    end do
  end do

  call hipCheck(hipMalloc(vel_dev, size(vel) * 8))
  call hipCheck(hipMalloc(dvel_dev, size(dvel) * 8))

  call hipCheck(hipMemcpy(vel_dev, c_loc(vel(1,1,1,1)), size(vel) * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dvel_dev, c_loc(dvel(1,1,1,1,1)), size(dvel) * 8, hipMemcpyHostToDevice))

  call hipCheck(hipMalloc(dx_dev, size(dx) * 8))
  call hipCheck(hipMalloc(dy_dev, size(dy) * 8))
  call hipCheck(hipMalloc(dz_dev, size(dz) * 8))

  blocks = dim3(8, 8, 8)
  grids = dim3(65, 65, 65)

  ind = 1

  call cpu_time(tstart)

  call gradcal_kernel(vel_dev, dvel_dev, dx_dev, ind)

  call hipCheck(hipDeviceSynchronize())

  call cpu_time(tfinish)
  ctime(1) = tfinish - tstart

  call hipCheck(hipMemcpy(c_loc(dvel(1,1,1,1,1)), dvel_dev, size(dvel) * 8, hipMemcpyDeviceToHost))

  print *, "Time Taken : ", ctime(1)
  print *, "vel  : ", vel(0:1,0:1,0:1,1)
  print *, "dvel : ", dvel(0:1,0:1,0:1,1,1)

  call hipCheck(hipFree(vel_dev))
  call hipCheck(hipFree(dvel_dev))

  deallocate(x)
  deallocate(vel)
  deallocate(dvel)
end program gpu_comp
