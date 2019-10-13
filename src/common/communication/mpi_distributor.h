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

#ifndef __SCFD_MPI_DISTRIBUTOR_H__
#define __SCFD_MPI_DISTRIBUTOR_H__

#include <vector>
#include <list>
#include <set>
#include <boost/foreach.hpp>
#include <boost/range/algorithm/sort.hpp>
#include <boost/range/algorithm/find.hpp>
#include <mpi.h>
#include <tensor_field/tensor_field_mem_funcs.h>
#include <tensor_field/t_tensor_field_tml.h>
#include <for_each/for_each_1d.h>
#include "comm.h"

//ATTENTION reqires mpi
//TODO case map.is_loc_glob_ind_order_preserv() == false is not done. however, i'm not sure we really need it for now
//TODO need some kind of common init interface which get graph in some form (may be through some interface or edges enumration)
//TODO copy just needed amount of data (according to actual current blocks_n)
//TODO for now i deleted variant with memcpy (cudaMemcpy) and leave only variant with copy kernels
//MPI_DISTRIBUTOR_USE_COPY_KERNELS macro was used to define behavoiur, we can restore second variant, if needed

#define MPI_DISTRIBUTOR_USE_COPY_KERNELS
//#define MPI_DISTRIBUTOR_USE_CUDA_AWARE_MPI

#include "detail/mpi_distributor.h"

namespace communication
{

//TODO in fact, in case of non-MPI_DISTRIBUTOR_USE_COPY_KERNELS we don't need ForEachType parameter
template<t_tensor_field_storage Storage, t_for_each_type ForEachType = FET_SERIAL_CPU>
struct mpi_distributor
{
    typedef detail::bucket                  bucket_t;
    typedef detail::packet_info<Storage>    packet_info_t;
    typedef detail::data_array_info         data_array_info_t;

    //this is supposed to be initied on some preface stage, contains global indexes
    std::set<int>                   stencil_elems;

    std::vector<data_array_info_t>  synced_data_info;

    //following data describes communication pattern
    std::list<packet_info_t>        packets_in, packets_out;
    std::vector<packet_info_t*>     packets_in_by_rank, packets_out_by_rank;
    int                             max_data_size_per_elem;
    
#ifdef MPI_DISTRIBUTOR_USE_COPY_KERNELS
    for_each_1d<ForEachType>        for_each;
#endif

    //NOTE in !packet_in case indices must already be in array (indices_glob)
    template<class Map>
    void    _init(const Map &map, packet_info_t &res, int rank_from, int rank_to, bool packet_in, std::vector<int> &indices_glob);

    void    update_buffers(int new_max_data_size_per_elem);

#ifdef MPI_DISTRIBUTOR_USE_COPY_KERNELS
    void    pack_package_dev(packet_info_t &pkg_out);
    void    unpack_package_dev(packet_info_t &pkg_in);
#else
#error "no copy kernels behavour is not realized"
#endif
    void    pack_package(packet_info_t &pkg_out)
    {
        pack_package_dev(pkg_out);
        if (pkg_out.is_buf_separate)
            tensor_field_copy_storage_2_host<Storage>(pkg_out.buf, pkg_out.dev_buf, pkg_out.buf_size);
    }
    void    unpack_package(packet_info_t &pkg_in)
    {
        if (pkg_in.is_buf_separate)
            tensor_field_copy_host_2_storage<Storage>(pkg_in.dev_buf, pkg_in.buf, pkg_in.buf_size);
        unpack_package_dev(pkg_in);
    }
    int             isend_requests_count, irecv_requests_count;
    MPI_Request     *isend_requests, *irecv_requests;

    void    isend(int to_rank, int &isend_requests_count, MPI_Request *isend_requests);
    void    irecv(int from_rank, int &irecv_requests_count, MPI_Request *irecv_requests);
    void    recv_finish(int from_rank);

public:
    mpi_distributor() : max_data_size_per_elem(0), isend_requests(NULL), irecv_requests(NULL)
    {
    }

    template<class Map>
    void    add_stencil_element(Map &map, int idx)
    {
        if (map.check_glob_owned(idx)) return;
        map.add_stencil_element(idx);
        stencil_elems.insert(idx);
    }
    template<class Map>
    void    add_stencil_element_pass_map(const Map &map, int idx)
    {
        if (map.check_glob_owned(idx)) return;
        stencil_elems.insert(idx);
    }
    template<class Map>
    void    init(const Map &map);

    void    start_set_data(int arrays_n = 1)
    {
        synced_data_info.reserve(arrays_n);
        synced_data_info.clear();
    }
    template<class T>
    void    add_data_array(T  *data_ptr, int data_sz, int blocks_n = 1, bool structure_of_blocks = true, int logic_io = 0)
    {
        synced_data_info.push_back( data_array_info_t((char*)data_ptr, sizeof(T), logic_io, data_sz, blocks_n, structure_of_blocks) );
    }
    void    stop_set_data()
    {
        //calc actual data_size_per_elem
        int     data_size_per_elem = 0;
        BOOST_FOREACH(data_array_info_t &array_info, synced_data_info) {
            data_size_per_elem += array_info.elem_size * array_info.blocks_n;
        }
        if (max_data_size_per_elem < data_size_per_elem) {
            printf("mpi_distributor stop_set_data: data_size_per_elem = %d is bigger then max_data_size_per_elem = %d -> reinit buffers\n", 
                data_size_per_elem, max_data_size_per_elem);
            update_buffers(data_size_per_elem);
        }
    }
    template<class T>
    void    set_data(T  *data_ptr, int data_sz, int blocks_n = 1, bool structure_of_blocks = true, int logic_io = 0)
    {
        start_set_data(1);
        add_data_array(data_ptr, data_sz, blocks_n, structure_of_blocks, logic_io);
        stop_set_data();   
    }
    //by agreement we suppose that logical enumeration in tensor fields to communicate is the one prescribed by Map
    //i think it's a good agreement. for example this fileds could actually be of different sizes, the just need to follow one local enumeration
    //(which is true almost anywhere)
    //TODO we here suppose some datail about tensor_fileds which could be changed:
    //There is no alignment inside tensor_fileds, i.e. t.N gives actuall stride size between subarrays
    template<class T>
    void    set_data(const t_tensor0_field_tml<T,Storage> &t)
    {
        set_data(t.d, t.N, 1, true, t.logic_i0);
    }
    template<class T,int d1,t_tensor_field_arrangement arrangement>
    void    set_data(const t_tensor1_field_tml<T,d1,Storage,arrangement> &t)
    {
        set_data(t.d, t.N, d1, arrangement == TFA_DEVICE_STYLE, t.logic_i0);
    }
    template<class T,int d1,int d2,t_tensor_field_arrangement arrangement>
    void    set_data(const t_tensor2_field_tml<T,d1,d2,Storage,arrangement> &t)
    {
        set_data(t.d, t.N, d1*d2, arrangement == TFA_DEVICE_STYLE, t.logic_i0);
    }
    template<class T,int d1,int d2,int d3,t_tensor_field_arrangement arrangement>
    void    set_data(const t_tensor3_field_tml<T,d1,d2,d3,Storage,arrangement> &t)
    {
        set_data(t.d, t.N, d1*d2*d3, arrangement == TFA_DEVICE_STYLE, t.logic_i0);
    }

    void    sync();

    ~mpi_distributor();
};

template<t_tensor_field_storage Storage, t_for_each_type ForEachType>
template<class Map>
void    mpi_distributor<Storage,ForEachType>::_init(const Map &map, packet_info_t &res, int rank_from, int rank_to, bool packet_in, std::vector<int> &indices_glob)
{
    //ISSUE hmmm, may be use different names 4 arrays (indices_glob and indices_loca)?? not important.

    //it's assert, but we can afford exception here
    if ((packet_in)&&(rank_to != map.get_own_rank())) throw std::logic_error("mpi_distributor::init: not own_rank is used");
    if ((!packet_in)&&(rank_from != map.get_own_rank())) throw std::logic_error("mpi_distributor::init: not own_rank is used");

    //NOTE here indices_glob are GLOBAL element indices
    if (packet_in) {
        indices_glob.clear();
        //logic checks in the begining of method mean that rank_to == map.get_own_rank(), because packet_in is true
        //that's why we can use stencil_elems wrt calling process
        BOOST_FOREACH(int idx, stencil_elems) {
            if ((idx != -1)&&(map.get_rank(idx) == rank_from)) indices_glob.push_back(idx);
        }
    }

    std::vector<int>        indices(indices_glob);

    //logic checks in the begining of method explain, why we can use local indexes wrt calling process in both cases
    //it's connected with the problem that Map concept does not allow to know local enumeration of 'foreign' process
    for (int i = 0;i < indices.size();++i) {
        indices[i] = map.glob2loc(indices[i]);
    }

    //here indices are LOCAL (WRT calling process) element indices
    boost::sort(indices);
    int i = 0;
    while (i < indices.size()) {
        res.buckets.push_back( bucket_t(indices[i],1) );
        while ((i+1 < indices.size())&&(indices[i+1] <= indices[i]+1)) { ++i; res.buckets.back().len = (indices[i] - res.buckets.back().i_start)+1; }
        ++i;
    }

    //FOR DEBUG
    /*printf("p=%d : rank_from = %d rank_to = %d\n", get_comm_rank(), rank_from, rank_to);
    BOOST_FOREACH(bucket_t &b, res.buckets) {
        printf("p=%d : bucket_start = %d bucket_len = %d\n", get_comm_rank(), b.i_start, b.len);
    }*/
}

template<t_tensor_field_storage Storage, t_for_each_type ForEachType>
template<class Map>
void    mpi_distributor<Storage,ForEachType>::init(const Map &map)
{
    int     comm_rank, comm_size;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor::init::MPI_Comm_rank failed");
    if (MPI_Comm_size(MPI_COMM_WORLD, &comm_size) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor::init::MPI_Comm_size failed");

    for (int diff_sz = -comm_size; diff_sz < comm_size;++diff_sz) {
        if (diff_sz == 0) continue;
         //init 'in' to the right
        std::vector<int>        indices_in_right, indices_out_left;
        if ((comm_rank + diff_sz >= 0)&&(comm_rank + diff_sz < comm_size)) {
            packets_in.push_back(packet_info_t());
            packets_in.back().proc_rank = comm_rank + diff_sz;
            packets_in.back().reorder_needed = !map.is_loc_glob_ind_order_preserv();
            _init(map, packets_in.back(), comm_rank + diff_sz, comm_rank, true, indices_in_right);
            //send indices size to the right
            int ind_n = indices_in_right.size();
            if (MPI_Send(&ind_n, sizeof(int), MPI_BYTE, comm_rank + diff_sz, 0, MPI_COMM_WORLD) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor_init:MPI_Send (ind_n) failed");
        }
        if ((comm_rank - diff_sz >= 0)&&(comm_rank - diff_sz < comm_size)) {
            //recv indices size from the left
            int ind_n_out_left;
            if (MPI_Recv(&ind_n_out_left, sizeof(int), MPI_BYTE, comm_rank - diff_sz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor_init:MPI_Recv (ind_n_out_left) failed");
            indices_out_left.resize(ind_n_out_left);
        }
        if ((comm_rank + diff_sz >= 0)&&(comm_rank + diff_sz < comm_size)) {
            //send indices to the right
            if (indices_in_right.size() > 0) {
                if (MPI_Send(&(indices_in_right.front()), sizeof(int)*indices_in_right.size(), MPI_BYTE, comm_rank + diff_sz, 0, MPI_COMM_WORLD) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor_init:MPI_Send (indices) failed");
            }
        }
        if ((comm_rank - diff_sz >= 0)&&(comm_rank - diff_sz < comm_size)) {
            //recv indices from the left
            if (indices_out_left.size() > 0) {
                if (MPI_Recv(&(indices_out_left.front()), sizeof(int)*indices_out_left.size(), MPI_BYTE, comm_rank - diff_sz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor_init:MPI_Recv (indices) failed");
            }
            //init 'out' to the left
            packets_out.push_back(packet_info_t());
            packets_out.back().proc_rank = comm_rank - diff_sz;
            packets_out.back().reorder_needed = !map.is_loc_glob_ind_order_preserv();
            _init(map, packets_out.back(), comm_rank, comm_rank - diff_sz, false, indices_out_left);
        }
    }

    BOOST_FOREACH(packet_info_t &pkg, packets_in) {
        pkg.init_copy_info();
    }
    BOOST_FOREACH(packet_info_t &pkg, packets_out) {
        pkg.init_copy_info();
    }

    //init packets_in_by_rank, packets_out_by_rank
    packets_in_by_rank.resize(comm_size, NULL);
    packets_out_by_rank.resize(comm_size, NULL);
    BOOST_FOREACH(packet_info_t &pkg, packets_in)
        packets_in_by_rank[pkg.proc_rank] = &pkg;
    BOOST_FOREACH(packet_info_t &pkg, packets_out)
        packets_out_by_rank[pkg.proc_rank] = &pkg;
}

template<t_tensor_field_storage Storage, t_for_each_type ForEachType>
void    mpi_distributor<Storage,ForEachType>::update_buffers(int new_max_data_size_per_elem)
{
    max_data_size_per_elem = new_max_data_size_per_elem;
    //reinit buffers
    BOOST_FOREACH(packet_info_t &pkg, packets_in) {
        pkg.update_buffer(max_data_size_per_elem);
    }
    BOOST_FOREACH(packet_info_t &pkg, packets_out) {
        pkg.update_buffer(max_data_size_per_elem);
    }
}

template<t_tensor_field_storage Storage, t_for_each_type ForEachType>
void    mpi_distributor<Storage,ForEachType>::isend(int to_rank, int &isend_requests_count, MPI_Request *isend_requests)
{
    if (packets_out_by_rank[to_rank] == NULL) return;
    packet_info_t &pkg = *packets_out_by_rank[to_rank];
    if (pkg.buf_size == 0) return;
    pack_package( pkg );
    if ( pkg.reorder_needed ) {
        throw std::logic_error("mpi_distributor::pkg.reorder_needed == true situation is not developed yet");
    }
    if (MPI_Isend(pkg.buf, pkg.buf_size, MPI_BYTE, to_rank, 0, MPI_COMM_WORLD, &(isend_requests[isend_requests_count])) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor::send::MPI_Send failed");
    isend_requests_count++;
}

template<t_tensor_field_storage Storage, t_for_each_type ForEachType>
void    mpi_distributor<Storage,ForEachType>::irecv(int from_rank, int &irecv_requests_count, MPI_Request *irecv_requests)
{
    if (packets_in_by_rank[from_rank] == NULL) return;
    packet_info_t &pkg = *packets_in_by_rank[from_rank];
    if (pkg.buf_size == 0) return;
    if (MPI_Irecv(pkg.buf, pkg.buf_size, MPI_BYTE, from_rank, 0, MPI_COMM_WORLD, &(irecv_requests[irecv_requests_count])) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor::send::MPI_Send failed");
    irecv_requests_count++;
}

//MPI_STATUS_IGNORE
template<t_tensor_field_storage Storage, t_for_each_type ForEachType>
void    mpi_distributor<Storage,ForEachType>::recv_finish(int from_rank)
{
    packet_info_t &pkg = *packets_in_by_rank[from_rank];
    if ( pkg.reorder_needed ) {
        throw std::logic_error("mpi_distributor::pkg.reorder_needed == true situation is not developed yet");
    }
    unpack_package( pkg );
}

template<t_tensor_field_storage Storage, t_for_each_type ForEachType>
void    mpi_distributor<Storage,ForEachType>::sync()
{
    int     comm_rank, comm_size;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor::sync::MPI_Comm_rank failed");
    if (MPI_Comm_size(MPI_COMM_WORLD, &comm_size) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor::sync::MPI_Comm_size failed");

    isend_requests_count = 0; irecv_requests_count = 0;
    //perhaps not the best place to do it.. but it's ok
    //ISSUE why new/delete for these buffers?
    if (isend_requests == NULL) isend_requests = new MPI_Request[comm_size];
    if (irecv_requests == NULL) irecv_requests = new MPI_Request[comm_size];

    //isend all
    for (int rank = 0; rank < comm_size;++rank) {
        if (rank == comm_rank) continue;
        isend(rank, isend_requests_count, isend_requests);
    }
    //irecv all
    for (int rank = 0; rank < comm_size;++rank) {
        if (rank == comm_rank) continue;
        irecv(rank, irecv_requests_count, irecv_requests);
    }
    //wait all irecv
    for (int ireq = 0;ireq < irecv_requests_count;++ireq) {
        int             ireq_idx;
        MPI_Status      status;
        if (MPI_Waitany( irecv_requests_count, irecv_requests, &ireq_idx, &status) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor::sync::MPI_Waitany failed");
        recv_finish( status.MPI_SOURCE );
    }
    //wait all isend
    if (MPI_Waitall( isend_requests_count, isend_requests, MPI_STATUS_IGNORE) != MPI_SUCCESS) throw std::runtime_error("mpi_distributor::sync::MPI_Waitall failed");
}

template<t_tensor_field_storage Storage, t_for_each_type ForEachType>
mpi_distributor<Storage,ForEachType>::~mpi_distributor()
{
    if (isend_requests != NULL) delete []isend_requests;
    if (irecv_requests != NULL) delete []irecv_requests;
}

}

#endif
