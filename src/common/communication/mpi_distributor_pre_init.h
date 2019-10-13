// Copyright © 2016 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_MPI_DISTRIBUTOR_INIT_H__
#define __SCFD_MPI_DISTRIBUTOR_INIT_H__

#include <mesh/t_cpu_mesh_tml.h>
#include <tensor_field/tensor_field_mem_funcs.h>
#include "mpi_distributor.h"

namespace communication
{

//ATTENTION reqires mpi
//TODO case map.is_loc_glob_ind_order_preserv() == false is not done. however, i'm not sure we really need it for now

template<class MAP,class MESH,class T,t_tensor_field_storage storage, t_for_each_type fet_type>
void    mpi_distributor_pre_init(MAP &map, mpi_distributor<T,storage,fet_type> &dist, MESH &cpu_mesh)
{
    int     comm_rank, comm_size;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor::init::MPI_Comm_rank failed");
    if (MPI_Comm_size(MPI_COMM_WORLD, &comm_size) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor::init::MPI_Comm_size failed");

    for (int i_ = 0;i_ < map.get_size();++i_) {
        int i = map.own_glob_ind(i_);
        #define PROCESS_INDEX(idx) if (((idx) != -1)&&(!map.check_glob_owned((idx)))) dist.add_stencil_element(map,(idx));
        for (int j = 0;j < cpu_mesh.cv[i].faces_n;++j) {
            PROCESS_INDEX(cpu_mesh.cv[i].neighbours[j]);
        }
        #undef PROCESS_INDEX
    }
}

}

#endif
