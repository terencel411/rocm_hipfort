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

  integer, parameter :: n = 128
  real :: tstart, tfinish, ctime(12)
  integer, parameter :: bytes_per_element = 4 ! integer
  integer(c_size_t), parameter :: Nbytes = n * bytes_per_element

  integer, allocatable, target, dimension(:) :: a, b
  integer :: n_dev = n

  integer :: i

  ctime = 0.0

  ! Allocate host memory
  allocate(a(n))
  allocate(b(n))

  ! Initialize host arrays
  a(:) = 2
  b(:) = 0

  ! Allocate array space on the device
  call hipCheck(hipMalloc(a_dev, Nbytes))
  call hipCheck(hipMalloc(b_dev, Nbytes))

  ! Transfer data from host to device memory
  ! c_loc returns the addr of the variable
  ! in this case mem location of the first ele in the array
  call hipCheck(hipMemcpy(a_dev, c_loc(a(1)), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(b_dev, c_loc(b(1)), Nbytes, hipMemcpyHostToDevice))

  call cpu_time(tstart)

  ! Launch the kernel
  call launch(a_dev, b_dev, n_dev)

  call hipCheck(hipDeviceSynchronize())

  call cpu_time(tfinish)
  ctime(1) = tfinish - tstart

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(c_loc(b(1)), b_dev, Nbytes, hipMemcpyDeviceToHost))

  ! Print results
  print *, ""
  ! print *, a(:32), a(n-5:)
  print *, "1-32", b(:32)
  print *, "33-64", b(33:64)
  print *, "65-98", b(65:98)
  print *, "99-128", b(99:128)

  call cpu_time(tstart)

  do i=1,n
    b(i) = a(i) + a(i)**2 + a(i)**3 + a(i)**4
  end do

  call cpu_time(tfinish)
  ctime(2) = tfinish - tstart

  print *, ""
  print *, "CPU : ", ctime(2)
  print *, "GPU : ", ctime(1)

  ! Free device memory
  call hipCheck(hipFree(a_dev))
  call hipCheck(hipFree(b_dev))

  ! Deallocate host memory
  deallocate(a)
  deallocate(b)

  write(*, *) "PASSED!"

end program fortran_hip
