// Copyright © 2016,2017 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_BLOCK_VECTOR_H__
#define __SCFD_BLOCK_VECTOR_H__

#define SCFD_BLOCK_VECTOR_ENABLE_MPI

#include <cassert>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <utils/cuda_safe_call.h>
#include <utils/device_tag.h>
#include <tensor_field/t_tensor_field_tml.h>
#ifdef SCFD_BLOCK_VECTOR_ENABLE_MPI
#include <communication/comm.h>
#endif

//ISSUE what about block sizes? are they static or dynamic?
//ISSUE can we construct it from tensor2_field if it would have dynamic dimensions
//TODO make it host compatible

namespace numerical_algos
{

template<class T,t_tensor_field_storage storage,class Map>
class block_vector
{
    int                                 block_size_;
    int                                 size_;
    t_tensor0_field_tml<T,storage>      vals_;

    //we simply forbit this operations for now
    block_vector(const block_vector &v) { } 
    block_vector &operator=(const block_vector &v) { return *this; }
public:
    block_vector() {}

    int                             size()const { return size_; }
    int                             block_size()const { return block_size_; }
    int                             total_size()const { return size_*block_size_; }
    T                               *ptr() { return vals_.d; }
    const T                         *ptr()const { return vals_.d; }
    t_tensor0_field_tml<T,storage>  array() { return vals_; }
    //TODO return const_tensor insted
    t_tensor0_field_tml<T,storage>  array()const { return vals_; }
    bool                            is_inited()const { return vals_.d != NULL; }

    void                            init(int block_size, int size)
    {
        assert(vals_.d == NULL);
        block_size_ = block_size;
        size_ = size;
        vals_.init(size_*block_size_);
    }
    void                            init(const Map &map, int block_size)
    {
        init(block_size, (map.max_loc_ind() - map.min_loc_ind())+1);
    }
    void                            init_from_file(const Map &map, const std::string &fn);
    void                            read_values_from_file(const Map &map, const std::string &fn);
    void                            write_to_file(const Map &map, const std::string &fn);

    /*~block_vector()
    {
        free();
    }*/
};

template<class T,t_tensor_field_storage storage,class Map>
void    block_vector<T,storage,Map>::init_from_file(const Map &map, const std::string &fn)
{
    std::ifstream f(fn.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("block_vector::init_from_file: error while opening file " + fn);
    
    int     algebraic_size, glob_size;
    
    if (!(f >> algebraic_size >> block_size_)) throw std::runtime_error("block_vector::init_from_file: error while reading sizes");

    if (algebraic_size%block_size_ != 0) throw std::runtime_error("block_vector::init_from_file: vector size is not divider of block size");
    glob_size = algebraic_size/block_size_;

    init(map, block_size_);

    t_tensor0_field_view_tml<T,storage>   vals_view(vals_, false);
    for (int i_glob = 0;i_glob < glob_size;++i_glob) {
        bool is_own = map.check_glob_owned(i_glob);
        int  i_loc;
        if (is_own) i_loc = map.glob2loc(i_glob); 
        for (int ii1 = 0;ii1 < block_size_;++ii1) {
            T    val;
            if (!(f >> val)) throw std::runtime_error("block_vector::init_from_file: error while reading values");
            if (is_own) vals_view(i_loc*block_size_ + ii1) = val;
        }
    }   
    vals_view.release(true); 
}

template<class T,t_tensor_field_storage storage,class Map>
void    block_vector<T,storage,Map>::read_values_from_file(const Map &map, const std::string &fn)
{
    std::ifstream f(fn.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("block_vector::read_values_from_file: error while opening file " + fn);
    
    int     algebraic_size, glob_size, block_size;
    
    if (!(f >> algebraic_size >> block_size)) throw std::runtime_error("block_vector::read_values_from_file: error while reading sizes");

    if (algebraic_size%block_size_ != 0) throw std::runtime_error("block_vector::read_values_from_file: vector size is not divider of block size");
    glob_size = algebraic_size/block_size;

    //TODO check sizes coincidence somehow

    t_tensor0_field_view_tml<T,storage>   vals_view(vals_, false);
    for (int i_glob = 0;i_glob < glob_size;++i_glob) {
        bool is_own = map.check_glob_owned(i_glob);
        int  i_loc;
        if (is_own) i_loc = map.glob2loc(i_glob); 
        for (int ii1 = 0;ii1 < block_size_;++ii1) {
            T    val;
            if (!(f >> val)) throw std::runtime_error("block_vector::read_values_from_file: error while reading values");
            if (is_own) vals_view(i_loc*block_size_ + ii1) = val;
        }
    }   
    vals_view.release(true); 
}

template<class T,t_tensor_field_storage storage,class Map>
void    block_vector<T,storage,Map>::write_to_file(const Map &map, const std::string &fn)
{
#ifdef SCFD_BLOCK_VECTOR_ENABLE_MPI
    int     ip = communication::get_comm_rank(),
            np = communication::get_comm_size();
#else
    int     ip = 0;
    int     np = 1;
#endif
    int     algebraic_size = map.get_total_size()*block_size_;
    if (ip == 0) {
        std::ofstream f(fn.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("block_vector::write_to_file: error while opening file " + fn);
        if (!(f << algebraic_size << " " << block_size_ << std::endl)) throw std::runtime_error("block_vector::write_to_file: error while writing sizes");
        f.close();
    }

    t_tensor0_field_view_tml<T,storage>   vals_view(vals_, true);
    for (int curr_ip = 0;curr_ip < np;++curr_ip) {

#ifdef SCFD_BLOCK_VECTOR_ENABLE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        if (ip != curr_ip) continue;

        std::ofstream f(fn.c_str(), std::ofstream::out | std::ofstream::app);
        if (!f) throw std::runtime_error("block_vector::write_to_file: error while opening file " + fn);

        f.setf(std::ios::scientific);
        f.precision(8);

        for(int i_ = 0;i_ < map.get_size();i_++) {
            int i_loc = map.own_loc_ind(i_);
            for (int ii1 = 0;ii1 < block_size_;++ii1) {
                T    val = vals_view(i_loc*block_size_ + ii1);
                if (!(f << val << std::endl)) throw std::runtime_error("block_vector::write_to_file: error while writing values");
            }
        }   

        f.close();
    }
    vals_view.release(false);
}

}

#endif
