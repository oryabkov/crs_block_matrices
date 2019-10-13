all: tensor_field_ndim_test_cpu tensor_field_ndim_test_gpu tensor_field_test_gpu

tensor_field_ndim_test_cpu:
		g++ -I.. tensor_field_ndim_test.cpp -o tensor_field_ndim_test_cpu
tensor_field_ndim_test_gpu:
		nvcc -I.. tensor_field_ndim_test.cu -o tensor_field_ndim_test_gpu
tensor_field_test_gpu:
		nvcc -I.. tensor_field_test.cu -o tensor_field_test_gpu