test_apply_single_gpu.bin: src/test_apply_single_gpu.cu
	nvcc -Isrc/common src/test_apply_single_gpu.cu -lcusparse -o test_apply_single_gpu.bin