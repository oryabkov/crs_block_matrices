-include config.inc
rect_map_test: rect_map_test.cpp
	g++ -I../.. rect_map_test.cpp -o rect_map_test
mpi_distributor_test_cpu: mpi_distributor_test.cu
	nvcc -arch=sm_35 -I../.. -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) -DTEST_CPU mpi_distributor_test.cu -o mpi_distributor_test_cpu -lmpi
mpi_distributor_test_cuda: mpi_distributor_test.cu
	nvcc -arch=sm_35 -I../.. -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) -DTEST_CUDA mpi_distributor_test.cu -o mpi_distributor_test_cuda -lmpi
mpi_distributor_test_cuda_aware: mpi_distributor_test.cu
	nvcc -arch=sm_35 -I../.. -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) -DTEST_CUDA -DMPI_DISTRIBUTOR_USE_CUDA_AWARE_MPI mpi_distributor_test.cu -o mpi_distributor_test_cuda_aware -lmpi