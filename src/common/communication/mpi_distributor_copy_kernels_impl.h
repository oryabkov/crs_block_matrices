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

#ifndef __SCFD_MPI_DISTRIBUTOR_COPY_KERNELS_IMPL_H__
#define __SCFD_MPI_DISTRIBUTOR_COPY_KERNELS_IMPL_H__

#include <for_each/for_each_1d.h>
//WARNING it's supposed that before this file inclusion there are inclusions of all needed for_each_1d_* implementation files; it's up to user of this class
/*TODO hmm??
#ifdef MCPU_MGPU_THERMOCOND_SOLVER_CUDA_NODE
#include "for_each_1d_cuda_impl.cuh"
#endif
#ifdef MCPU_MGPU_THERMOCOND_SOLVER_OPENMP_NODE
#include "for_each_1d_openmp_impl.h"
#endif*/
#include <for_each/for_each_func_macro.h>
#include "mpi_distributor.h"

namespace communication
{

#ifdef MPI_DISTRIBUTOR_USE_COPY_KERNELS

//TODO structure_of_blocks cases should be done as two different cases with differnt order (iblock = (ith/data_blocks_n) and etc)
//TODO structure_of_blocks should be reflected in buf ordering
//ISSUE maybe make elem_size as special case?

template<class T>
struct mpi_distributor_pack_pkg_func
{
    //elem_size here in sizeof(T)
    FOR_EACH_FUNC_PARAMS_HELP(mpi_distributor_pack_pkg_func,
                              int, elem_size, int, buf_block_size, T*, buf, 
                              int, data_blocks_n, int, data_sz, int, logic_io, bool, structure_of_blocks, T*, data,
                              int*, ind)

    __DEVICE_TAG__ void operator()(const int &ith)const
    {
        int iblock = (ith/buf_block_size),
            i = (ith%buf_block_size);
        if (!(iblock < data_blocks_n)) return;
        if (!(i < buf_block_size)) return;
        if (structure_of_blocks) {
            for (int ii = 0;ii < elem_size;++ii)
                buf[(buf_block_size*iblock+i)*elem_size + ii] = data[(data_sz*iblock + ind[i]-logic_io)*elem_size + ii];
        } else {
            for (int ii = 0;ii < elem_size;++ii)
                buf[(buf_block_size*iblock+i)*elem_size + ii] = data[((ind[i]-logic_io)*data_blocks_n + iblock)*elem_size + ii];
        }
    }

};

template<class T>
struct mpi_distributor_unpack_pkg_func
{
    //elem_size here in sizeof(T)
    FOR_EACH_FUNC_PARAMS_HELP(mpi_distributor_unpack_pkg_func,
                              int, elem_size, int, buf_block_size, T*, buf, 
                              int, data_blocks_n, int, data_sz, int, logic_io, bool, structure_of_blocks, T*, data,
                              int*, ind)

    __DEVICE_TAG__ void operator()(const int &ith)const
    {
        int iblock = (ith/buf_block_size),
            i = (ith%buf_block_size);
        if (!(iblock < data_blocks_n)) return;
        if (!(i < buf_block_size)) return;
        if (structure_of_blocks) {
            for (int ii = 0;ii < elem_size;++ii)
                data[(data_sz*iblock + ind[i]-logic_io)*elem_size + ii] = buf[(buf_block_size*iblock+i)*elem_size + ii];
        } else {
            for (int ii = 0;ii < elem_size;++ii)
                data[((ind[i]-logic_io)*data_blocks_n + iblock)*elem_size + ii] = buf[(buf_block_size*iblock+i)*elem_size + ii];
        }
    }

};

template<t_tensor_field_storage storage, t_for_each_type fet_type>
void    mpi_distributor<storage,fet_type>::pack_package_dev(packet_info_t &pkg_out)
{
    if (pkg_out.buf_block_size == 0) return;

    int offset = 0;
    for (int i = 0;i < synced_data_info.size();++i) {
        const data_array_info_t   &info = synced_data_info[i];

        #define MPI_DISTRIBUTOR__PACK__(T)    \
            for_each( mpi_distributor_pack_pkg_func<T>(   info.elem_size/sizeof(T), pkg_out.buf_block_size, (T*)(pkg_out.dev_buf + offset),     \
                                                          info.blocks_n, info.data_sz, info.logic_io, info.structure_of_blocks, (T*)info.data,  \
                                                          pkg_out.dev_ind),                                                                     \
                    0, info.blocks_n*pkg_out.buf_block_size )

        if (info.elem_size % sizeof(long) == 0 && offset % sizeof(long) == 0) 
            MPI_DISTRIBUTOR__PACK__(long);
        else if (info.elem_size % sizeof(int) == 0 && offset % sizeof(int) == 0) 
            MPI_DISTRIBUTOR__PACK__(int);
        else if (info.elem_size % sizeof(short) == 0 && offset % sizeof(short) == 0) 
            MPI_DISTRIBUTOR__PACK__(short);
        else 
            MPI_DISTRIBUTOR__PACK__(char);

        #undef MPI_DISTRIBUTOR__PACK__

        offset += info.elem_size*pkg_out.buf_block_size*info.blocks_n;
    }
    //TODO what about this MPI_DISTRIBUTOR_USE_CUDA_AWARE_MPI check?
    //Can we leave this wait() call for all cases? (seems we can)
#ifdef MPI_DISTRIBUTOR_USE_CUDA_AWARE_MPI
    for_each.wait();
#endif
}

template<t_tensor_field_storage storage, t_for_each_type fet_type>
void    mpi_distributor<storage,fet_type>::unpack_package_dev(packet_info_t &pkg_in)
{
    if (pkg_in.buf_block_size == 0) return;

    int offset = 0;
    for (int i = 0;i < synced_data_info.size();++i) {
        const data_array_info_t   &info = synced_data_info[i];

        #define MPI_DISTRIBUTOR__UNPACK__(T)    \
            for_each( mpi_distributor_unpack_pkg_func<T>( info.elem_size/sizeof(T), pkg_in.buf_block_size, (T*)(pkg_in.dev_buf + offset),       \
                                                          info.blocks_n, info.data_sz, info.logic_io, info.structure_of_blocks, (T*)info.data,  \
                                                          pkg_in.dev_ind),                                                                      \
                    0, info.blocks_n*pkg_in.buf_block_size )

        if (info.elem_size % sizeof(long) == 0 && offset % sizeof(long) == 0) 
            MPI_DISTRIBUTOR__UNPACK__(long);
        else if (info.elem_size % sizeof(int) == 0 && offset % sizeof(int) == 0) 
            MPI_DISTRIBUTOR__UNPACK__(int);
        else if (info.elem_size % sizeof(short) == 0 && offset % sizeof(short) == 0) 
            MPI_DISTRIBUTOR__UNPACK__(short);
        else 
            MPI_DISTRIBUTOR__UNPACK__(char); 
            
        #undef MPI_DISTRIBUTOR__UNPACK__

        offset += info.elem_size*pkg_in.buf_block_size*info.blocks_n;
    }
}

#endif

}

#endif
