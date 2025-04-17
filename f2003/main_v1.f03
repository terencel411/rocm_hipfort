program fortran_hip
  use iso_c_binding
  use hipfort
  use hipfort_check

  implicit none

  interface
     subroutine launch(out,a,b,N) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr) :: a, b, out
       integer, value :: N
     end subroutine
  end interface

  type(c_ptr) :: da = c_null_ptr
  type(c_ptr) :: db = c_null_ptr
  type(c_ptr) :: dout = c_null_ptr

  integer, parameter :: N = 1000000
  integer, parameter :: bytes_per_element = 8 !double precision
  integer(c_size_t), parameter :: Nbytes = N*bytes_per_element

  ! Plain real should be equivalent to float
  real(8),allocatable,target,dimension(:) :: a, b, out
  real(8) :: error
  real(8), parameter :: error_max = 1.0d-10
  !  
  integer(c_size_t) :: size_var
  !  
  integer :: i
  type(hipDeviceProp_t),target :: props
  !
  call hipCheck(hipGetDeviceProperties(props,0))  
  write(*,"(a)",advance="no") "-- Running test 'vecadd' (Fortran 2003 interfaces)"
  write(*,"(a)",advance="no") "- device: "
  ! write(*,"(a)",advance="no") props%name
  
  i=1
  do while ( iachar(props%name(i)) .ne. 0 ) ! print till end char
    write(*,"(a)",advance="no") props%name(i)
    i = i+1
  end do 
  ! write(*,"(a)",advance="no") " - "

  ! Allocate host memory
  allocate(a(N))
  allocate(b(N))
  allocate(out(N))

  size_var = 8 * size(a)

  ! Initialize host arrays
  a(:) = 1.0
  b(:) = 2.0
  ! c(:) = 0.0

  ! Allocate array space on the device
  call hipCheck(hipMalloc(da,size_var))
  call hipCheck(hipMalloc(db,size_var))
  call hipCheck(hipMalloc(dout,size_var))

  ! Transfer data from host to device memory
  call hipCheck(hipMemcpy(da, c_loc(a(1)), size_var, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(db, c_loc(b(1)), size_var, hipMemcpyHostToDevice))

  call launch(dout,da,db,N)

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(c_loc(out(1)), dout, size_var, hipMemcpyDeviceToHost))
  
  print *, ""
  print *, a(:5)
  print *, b(:5)
  print *, out(:5)

  ! Verification
  do i = 1,N
     error = abs(out(i) - (a(i)+b(i)) )
     if( error .gt. error_max ) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " Out = ", out(i), i
        call exit
     endif
  end do

  call hipCheck(hipFree(da))
  call hipCheck(hipFree(db))
  call hipCheck(hipFree(dout))
  
  ! Deallocate host memory
  deallocate(a)
  deallocate(b)
  deallocate(out)

  write(*,*) "EXAMPLE CASE PASSED!"

end program fortran_hip
