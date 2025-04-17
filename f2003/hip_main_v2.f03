module vars
  use iso_c_binding
  use hipfort
  use hipfort_check
  implicit none

  integer(c_int), parameter :: im=16, jm=16, km=16, hm=5
  integer(c_int), parameter :: im_dev=16, jm_dev=16, km_dev=16, hm_dev=5

  real(8), parameter :: pi=4.d0*atan(1.0_8), num1d60=1.d0/60.d0

  type(dim3) :: grids, blocks
  integer :: istat
  character(len=100) :: error_string
  real :: tstart, tfinish, ctime(12)
  
  real(8), target :: dx, dy, dz
  real(8), allocatable, target, dimension(:,:,:,:) :: x, vel
  real(8), allocatable, target, dimension(:,:,:,:,:) :: dvel

  type(c_ptr) :: vel_dev = c_null_ptr
  type(c_ptr) :: dvel_dev = c_null_ptr
  type(c_ptr) :: dx_dev, dy_dev, dz_dev
  integer(c_int) :: ind

  integer(c_size_t), parameter :: bytes_per_element = 8 ! for real(8)
  integer(c_size_t) :: Nbytes, Nbytes_4, Nbytes_5
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
       type(c_ptr) :: dx_dev
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
  integer(c_int) :: err

  allocate(x(0:im,0:jm,0:km,1:3))
  allocate(vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:3))
  allocate(dvel(0:im,0:jm,0:km,1:3,1:3))

  Nbytes = bytes_per_element
  Nbytes_4 = size(vel) * bytes_per_element
  Nbytes_5 = size(dvel) * bytes_per_element

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

  ! to use c_loc for the vars, the vars need to have 'target' attr

  call hipCheck(hipMalloc(vel_dev, Nbytes_4))
  call hipCheck(hipMalloc(dvel_dev, Nbytes_5))

  call hipCheck(hipMemcpy(vel_dev, c_loc(vel(1,1,1,1)), Nbytes_4, hipMemcpyHostToDevice))
  
  call hipCheck(hipMemcpy(dx_dev, c_loc(dx), Nbytes, hipMemcpyHostToDevice))

  ind = 1

  call cpu_time(tstart)

  call gradcal_kernel(vel_dev, dvel_dev, dx_dev, ind)

  err = hipGetLastError()
  print *, "Error : ", err
  ! if (err /= hipSuccess) then
  !   print *, "Kernel launch failed: ", hipGetErrorString(err)
  ! end if

  call hipCheck(hipDeviceSynchronize())

  call cpu_time(tfinish)
  ctime(1) = tfinish - tstart

  call hipCheck(hipMemcpy(c_loc(dvel(1,1,1,1,1)), dvel_dev, Nbytes_5, hipMemcpyDeviceToHost))

  print *, "Time Taken : ", ctime(1)
  print *, "vel  : ", vel(0:1,0:1,0:1, 1)
  print *, "dvel : ", dvel(0:1,0:1,0:1,1,1)

  call hipCheck(hipFree(vel_dev))
  call hipCheck(hipFree(dvel_dev))

  deallocate(x)
  deallocate(vel)
  deallocate(dvel)

  print *, "Complete!"
end program gpu_comp
