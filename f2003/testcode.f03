program testcode
!
use hipfort
!
implicit none
!
integer :: ierr
real, dimension(:), allocatable :: a_h, b_h, c_h
real, dimension(:), pointer :: a_d, b_d, c_d

! Allocate host arrays
allocate(a_h(100), b_h(100), c_h(100))
a_h = 1.0
b_h = 2.0

! Allocate device arrays
ierr = hipMalloc(a_d, size(a_h))
ierr = hipMalloc(b_d, size(b_h))
ierr = hipMalloc(c_d, size(c_h))

! Copy data to device
ierr = hipMemcpy(a_d, a_h, size(a_h), hipMemcpyHostToDevice)
ierr = hipMemcpy(b_d, b_h, size(b_h), hipMemcpyHostToDevice)

! Perform vector addition on the device
call hipblasSaxpy(100, 1.0, a_d, 1, b_d, 1)

! Copy result back to host
ierr = hipMemcpy(c_h, b_d, size(b_h), hipMemcpyDeviceToHost)

! Free device memory
ierr = hipFree(a_d)
ierr = hipFree(b_d)
ierr = hipFree(c_d)

end program testcode
