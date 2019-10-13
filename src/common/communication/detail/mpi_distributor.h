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

#ifndef __SCFD_MPI_DISTRIBUTOR_DETAIL_H__
#define __SCFD_MPI_DISTRIBUTOR_DETAIL_H__

#include <vector>
#include <list>
#include <set>
#include <boost/foreach.hpp>
#include <boost/range/algorithm/sort.hpp>
#include <boost/range/algorithm/find.hpp>
#include <mpi.h>
//#include <cuda_ownutils.h>
#include <tensor_field/tensor_field_mem_funcs.h>
#include <tensor_field/t_tensor_field_tml.h>
#include <for_each/for_each_1d.h>

namespace communication
{
namespace detail
{

//by agreement i_start here is index in local enumeration (according to MAP)
struct bucket
{
    int     i_start, len;
    bucket() {}
    bucket(int _i_start,int _len) : i_start(_i_start), len(_len) { }
};

template<t_tensor_field_storage Storage>
struct packet_info
{
    int                     proc_rank;
    std::vector<bucket>     buckets;
    int                     buf_block_size;
    int                     max_data_size_per_elem; //in bytes
    int                     buf_size;               //in bytes
    bool                    is_buf_separate;        //if is_buf_separate, then buf is peparate host buffer
                                                    //otherwise buf == dev_buf
    char                    *dev_buf;               //device buffer
    char                    *buf;                   //cpu buffer
    bool                    reorder_needed;
    //TODO
    //this used only when MPI_DISTRIBUTOR_USE_COPY_KERNELS is endabled
    //dev_ind has size buf_block_size
    int                     *dev_ind;       //for each element of one data buffer block logic data index has to be copied is stored

    packet_info() : max_data_size_per_elem(0), buf_size(0), is_buf_separate(true), dev_buf(NULL), buf(NULL), dev_ind(NULL)
    {
        if (Storage == TFS_HOST) is_buf_separate = false;
#ifdef MPI_DISTRIBUTOR_USE_CUDA_AWARE_MPI
        is_buf_separate = false;
#endif        
    }
    ~packet_info()
    {
        if (dev_ind != NULL) tensor_field_free<Storage>(dev_ind);
        if (is_buf_separate && (buf != NULL)) tensor_field_free<TFS_HOST>(buf);
        if (dev_buf != NULL) tensor_field_free<Storage>(dev_buf);
    }

    void    init_block_size_()
    {
        buf_block_size = 0;
        BOOST_FOREACH(bucket &b, buckets) buf_block_size += b.len;
    }
#ifdef MPI_DISTRIBUTOR_USE_COPY_KERNELS
    void    init_copy_info()
    {
        init_block_size_();

        if (buf_block_size == 0) return;

        int     *dev_ind_host;

        tensor_field_malloc<Storage>((void**)&dev_ind, sizeof(int)*buf_block_size);
        //CUDA_SAFE_CALL( cudaMalloc((void**)&dev_ind, sizeof(int)*buf_block_size) );
        tensor_field_malloc<TFS_HOST>((void**)&dev_ind_host, sizeof(int)*buf_block_size);
        //CUDA_SAFE_CALL( cudaMallocHost((void**)&dev_ind_host, sizeof(int)*buf_block_size) );

        int     dev_ind_i = 0;
        BOOST_FOREACH(bucket &b, buckets) {
            for (int i = 0;i < b.len;++i) {
                dev_ind_host[dev_ind_i] = b.i_start + i;
                dev_ind_i++;
            }
        }

        tensor_field_copy_host_2_storage<Storage>(dev_ind, dev_ind_host, sizeof(int)*buf_block_size);
        //CUDA_SAFE_CALL( cudaMemcpy(dev_ind, dev_ind_host, sizeof(int)*buf_block_size, cudaMemcpyHostToDevice) );
        tensor_field_free<TFS_HOST>(dev_ind_host);
        //CUDA_SAFE_CALL( cudaFreeHost(dev_ind_host) );
    }
#else
#error "no copy kernels behavour is not realized"
    void    init_copy_info()
    {
        init_block_size_();
    }
#endif
    void    update_buffer(int max_data_size_per_elem_)
    {
        if (dev_buf != NULL) {
            tensor_field_free<Storage>(dev_buf); dev_buf = NULL;
        }
        if (is_buf_separate && (buf != NULL)) {
            tensor_field_free<TFS_HOST>(buf); buf = NULL;
        }

        max_data_size_per_elem = max_data_size_per_elem_;
        buf_size = max_data_size_per_elem*buf_block_size;
        if (buf_size > 0) {
            tensor_field_malloc<Storage>( (void**)&(dev_buf), buf_size );
            if (is_buf_separate) 
                tensor_field_malloc<TFS_HOST>( (void**)&(buf), buf_size );
            else 
                buf = dev_buf;
        }
    }
};

struct data_array_info
{
    //pointer to device data
    char                    *data;
    int                     elem_size;              //in fact, it is sizeof(T)
    int                     logic_io;
    int                     data_sz;
    int                     blocks_n;               //data consists of blocks_n parts each of size data_sz*elem_sz
    bool                    structure_of_blocks;    //

    data_array_info()
    {
    }
    data_array_info(char  *data_, int elem_size_, int logic_io_, int  data_sz_, 
                    int blocks_n_, bool structure_of_blocks_) : 
        data(data_), elem_size(elem_size_), logic_io(logic_io_), data_sz(data_sz_), 
        blocks_n(blocks_n_), structure_of_blocks(structure_of_blocks_)
    {
    }
};

}
}

#endif

