all: for_each_ndim_test

for_each_ndim_test: for_each_ndim_test.cu
		nvcc -I.. for_each_ndim_test.cu -o for_each_ndim_test
