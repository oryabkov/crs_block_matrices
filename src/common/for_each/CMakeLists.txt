

set (SOURCEFILES for_each_ndim_test.cu)
set (HEADERFILES for_each_ndim_openmp_impl.h for_each_enums.h for_each_ndim_cuda_impl.cuh for_each_storage_types.h for_each_ndim.h)

cuda_add_executable( for_each_ndim_test for_each_ndim_test.cu )

add_test(       NAME for_each_ndim_test
                COMMAND ./for_each_ndim_test)
