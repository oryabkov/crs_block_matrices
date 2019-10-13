-include config.inc
test_lu_sgs_float: test_lu_sgs_cuda.cu
	nvcc -arch=sm_35 -O3 -I../.. -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) -DSCALAR_TYPE=float test_lu_sgs_cuda.cu -lcusparse -lmpi -o test_lu_sgs_float
test_lu_sgs_double: test_lu_sgs_cuda.cu
	nvcc -arch=sm_35 -O3 -I../.. -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) -DSCALAR_TYPE=double test_lu_sgs_cuda.cu -lcusparse -lmpi -o test_lu_sgs_double

test_lu_sgs_cuda_aware_mpi_float: test_lu_sgs_cuda.cu
	nvcc -arch=sm_35 -O3 -I../.. -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) -DSCALAR_TYPE=float -DMPI_DISTRIBUTOR_USE_CUDA_AWARE_MPI test_lu_sgs_cuda.cu -lcusparse -lmpi -o test_lu_sgs_cuda_aware_mpi_float
test_lu_sgs_cuda_aware_mpi_double: test_lu_sgs_cuda.cu
	nvcc -arch=sm_35 -O3 -I../.. -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) -DSCALAR_TYPE=double -DMPI_DISTRIBUTOR_USE_CUDA_AWARE_MPI test_lu_sgs_cuda.cu -lcusparse -lmpi -o test_lu_sgs_cuda_aware_mpi_double

test_scalar_prod_float: test_scalar_prod.cu
	nvcc -arch=sm_35 -O3 -I../.. -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) -DSCALAR_TYPE=float test_scalar_prod.cu -lcusparse -lmpi -o test_scalar_prod_float