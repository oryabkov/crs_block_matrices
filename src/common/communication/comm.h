#ifndef __SCFD_COMM_H__
#define __SCFD_COMM_H__

#include <mpi.h>
#include <stdexcept>

namespace communication
{

inline int      get_comm_rank()
{
    int     comm_rank;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank) != MPI_SUCCESS) throw std::runtime_error("get_comm_rank()::MPI_Comm_rank failed");
    return comm_rank;
}

inline int      get_comm_size()
{
    int     comm_size;
    if (MPI_Comm_size(MPI_COMM_WORLD, &comm_size) != MPI_SUCCESS) throw std::runtime_error("get_comm_size()::MPI_Comm_size failed");
    return comm_size;
}

}

#endif
