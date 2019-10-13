-include config.inc
test_apply: test_apply.cu
	nvcc -arch=sm_35 -I../../.. -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) test_apply.cu -lcusparse -lmpi -o test_apply