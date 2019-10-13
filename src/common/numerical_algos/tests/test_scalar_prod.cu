

#include <cstdlib>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#include <utils/init_cuda.h>
#include <utils/log_mpi.h>
#include <for_each/for_each_1d.h>
#include <for_each/for_each_1d_cuda_impl.cuh>
#include <for_each/for_each_func_macro.h>
#include <communication/comm.h>
#include <communication/linear_partitioner.h>
#include <communication/parted_map.h>
#include <numerical_algos/vectors/block_vector.h>
#include <numerical_algos/vectors/block_vector_operations.h>

typedef SCALAR_TYPE                                                     real;
typedef utils::log_mpi                                                  log_t;
static const t_tensor_field_storage                                     storage = TFS_DEVICE;
static const t_for_each_type                                            for_each_type = FET_CUDA;
typedef for_each_1d<for_each_type>                                      for_each_t;
typedef communication::linear_partitioner                               partitioner_t;
typedef communication::parted_map<partitioner_t>                        map_t;
typedef numerical_algos::block_vector<real,storage,map_t>               vector_t;
typedef numerical_algos::block_vector_operations<real,storage,map_t>    vector_operations_t;

int read_vector_size(const std::string &fn)
{
    std::ifstream f(fn.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_vector_size: error while opening file " + fn);
    int     algebraic_size, block_size;
    if (!(f >> algebraic_size >> block_size)) throw std::runtime_error("read_vector_size: error while reading sizes");
    f.close();

    return algebraic_size/block_size;
}

int main(int argc, char **args)
{
    if (MPI_Init(&argc, &args) != MPI_SUCCESS) {
        std::cout << "ERROR: MPI_Init call failed ; abort" << std::endl;
        return 1;
    }

    int     comm_rank = communication::get_comm_rank(),
            comm_size = communication::get_comm_size();
    log_t   log;

    if (argc < 3) {
        if (comm_rank == 0)
            std::cout << "USAGE: " << std::string(args[0]) << " <vec1_fn> <vec2_fn>" << std::endl;
        if (MPI_Finalize() != MPI_SUCCESS) {
            std::cout << "WARNING: MPI_Finalize call failed" << std::endl;
            return 2;
        }
        return 0;
    }
    std::string             vec1_fn(args[1]), vec2_fn(args[2]);

    //TODO device number
    utils::init_cuda(1+comm_rank);

    log.info_f("creating map and partitioner...");
    int                     glob_size = read_vector_size(vec1_fn);
    partitioner_t           partitioner(glob_size, comm_size, comm_rank);
    map_t                   map(partitioner, true);

    map.complete();

    vector_t                vec1, vec2, w;

    log.info_f("reading vec1 from file %s...", vec1_fn.c_str());
    vec1.init_from_file(map, vec1_fn);
    log.info_f("reading vec2 from file %s...", vec2_fn.c_str());
    vec2.init_from_file(map, vec2_fn);
    
    vector_operations_t     vec_ops(&map, vec1.block_size());
    log.info_f("initializing scalar_prod weights...");
    w.init(map, vec1.block_size());
    vec_ops.assign_scalar(real(1.f), w);
    vec_ops.set_scalar_prod_weights(w);

    log.info_f("(vec1,vec2) = %e", vec_ops.scalar_prod(vec1, vec2));
 
    if (MPI_Finalize() != MPI_SUCCESS) {
        std::cout << "WARNING: MPI_Finalize call failed" << std::endl;
        return 3;
    }

    return 0;
}