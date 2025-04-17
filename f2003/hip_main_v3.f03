program fortran_hip
  use iso_c_binding
  use hipfort
  use hipfort_check

  implicit none

  interface
     subroutine launch(a_dev, b_dev, n_dev) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr) :: a_dev, b_dev
       integer, value :: n_dev
     end subroutine
  end interface

  type(c_ptr) :: a_dev = c_null_ptr
  type(c_ptr) :: b_dev = c_null_ptr

  integer(c_int) :: err

  integer, parameter :: n = 8
  real :: tstart, tfinish, ctime(12)
  real(8), allocatable, target, dimension(:,:) :: a, b
  integer :: n_dev = n
  integer :: i, j

  integer(c_size_t), parameter :: bytes_per_element = 8 ! for real(8)
  integer(c_size_t) :: Nbytes

  ctime = 0.0

  allocate(a(n,n))
  allocate(b(n,n))

  ! Initialize host arrays
  a(:,:) = 2
  b(:,:) = 0

  Nbytes = size(a) * bytes_per_element

  call hipCheck(hipMalloc(a_dev, Nbytes))
  call hipCheck(hipMalloc(b_dev, Nbytes))

  call hipCheck(hipMemcpy(a_dev, c_loc(a(1,1)), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(b_dev, c_loc(b(1,1)), Nbytes, hipMemcpyHostToDevice))

  call cpu_time(tstart)

  ! Launch the kernel
  call launch(a_dev, b_dev, n_dev)

  ! err = hipPeekAtLastError()
  ! if (err /= hipSuccess) then
  !   print *, "Kernel launch failed: ", hipGetErrorString(err)
  ! end if

  call hipCheck(hipDeviceSynchronize())

  call cpu_time(tfinish)
  ctime(1) = tfinish - tstart

  call hipCheck(hipMemcpy(c_loc(b(1,1)), b_dev, Nbytes, hipMemcpyDeviceToHost))
  
  print *, " "
  print *, "b(:,1) : ", b(:,1)
  print *, "b(:,2) : ", b(:,2)
  print *, "b(:,3) : ", b(:,3)
  print *, "b(:,4) : ", b(:,4)
  print *, "b(:,5) : ", b(:,5)
  print *, "b(:,6) : ", b(:,6)
  print *, "b(:,7) : ", b(:,7)
  print *, "b(:,8) : ", b(:,8)
  call cpu_time(tstart)

  do j = 1, n
    do i = 1, n
      b(i,j) = a(i,j) + a(i,j)**2
    end do
  end do

  call cpu_time(tfinish)
  ctime(2) = tfinish - tstart

  print *, ""
  print *, "CPU : ", ctime(2)
  print *, "GPU : ", ctime(1)

  call hipCheck(hipFree(a_dev))
  call hipCheck(hipFree(b_dev))

  deallocate(a)
  deallocate(b)

end program fortran_hip
