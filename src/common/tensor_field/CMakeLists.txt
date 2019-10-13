
set (SOURCEFILES tensor_field_ndim_test.cpp tensor_field_ndim_test.cu tensor_field_test.cu)
set (HEADERFILES t_tensor_field_ndim_tml.h t_tensor_field_tml.h tensor_field_enums.h tensor_field_mem_funcs.h t_rect_tml.h)

add_executable( tensor_field_ndim_test_cpu tensor_field_ndim_test.cpp )
cuda_add_executable( tensor_field_ndim_test_gpu tensor_field_ndim_test.cu )
cuda_add_executable( tensor_field_test_gpu tensor_field_test.cu )

add_test(       NAME tensor_field_ndim_test_cpu COMMAND ./tensor_field_ndim_test_cpu)
add_test(       NAME tensor_field_ndim_test_gpu COMMAND ./tensor_field_ndim_test_gpu)
add_test(       NAME tensor_field_test_gpu 	COMMAND ./tensor_field_test_gpu)