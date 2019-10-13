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

#ifndef __SCFD_COLOR_FILE_PARTITIONER_H__
#define __SCFD_COLOR_FILE_PARTITIONER_H__

#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include "linear_partitioner.h"

namespace communication
{

//TODO color_file_partitioner has pretty ugly realised feature, allowing it to fall back to linear_partitioner, if there is no file name specified
//it's usefull but pretty ugly - think about

//supposed to satisfy PARTITIONER concept

struct color_file_partitioner
{
    int                     total_size;
    int                     my_rank;
    std::string             f_name;
    bool                    is_complete;
    //first int is global index of element, second int is rank
    std::map<int,int>       ranks;
    //supposed to be sorted
    std::vector<int>        own_glob_indices;
    std::map<int,int>       own_glob_indices_2_ind;

    bool                    use_linear_partitioner;
    linear_partitioner    linear_partitioner;

    color_file_partitioner() {}
    color_file_partitioner(int N, int comm_size, int _my_rank, const std::string &_f_name) : use_linear_partitioner(false), f_name(_f_name), is_complete(false)
    {
        //total_size = N;
        my_rank = _my_rank;
        //TODO use c++ file streams (they are better for case of errors, because they close file in destructor)
        FILE    *f = fopen( f_name.c_str(), "rb" );
        if (f == NULL) throw std::runtime_error("color_file_partitioner::color_file_partitioner: failed to open file " + f_name);
        if (fread(&total_size, sizeof(total_size), 1, f) != 1) throw std::runtime_error("color_file_partitioner::color_file_partitioner: error while reading file " + f_name);
        if (total_size != N) throw std::runtime_error("color_file_partitioner::color_file_partitioner: error: mesh size in part file " + f_name + " differs from actual mesh size");
        int     file_colors_n;
        if (fread(&file_colors_n, sizeof(file_colors_n), 1, f) != 1) throw std::runtime_error("color_file_partitioner::color_file_partitioner: error while reading file " + f_name);
        if (file_colors_n != comm_size) throw std::runtime_error("color_file_partitioner::color_file_partitioner: error: colors number in part file " + f_name + " differs from actual communicator size");
        for (int i = 0;i < total_size;++i) {
            int color;
            if (fread(&color, sizeof(color), 1, f) != 1) throw std::runtime_error("color_file_partitioner::color_file_partitioner: error while reading file " + f_name);
            if (color == my_rank) {
                ranks[i] = color;
                own_glob_indices_2_ind[i] = own_glob_indices.size();
                own_glob_indices.push_back(i);
            }
        }
        fclose(f);
    }
    color_file_partitioner(int N, int comm_size, int _my_rank) : use_linear_partitioner(true), linear_partitioner(N, comm_size, _my_rank)
    {
        total_size = N;
        my_rank = _my_rank;
        for (int i = 0;i < total_size;++i) {
            if (linear_partitioner.check_glob_owned(i)) {
                ranks[i] = my_rank;
                own_glob_indices_2_ind[i] = own_glob_indices.size();
                own_glob_indices.push_back(i);
            }
        }
    }

    //'read' before construction part:
    int     get_own_rank()const { return my_rank; }
    //i_glob here could be any global index both before and after construction part
    bool    check_glob_owned(int i_glob)const 
    {
        std::map<int,int>::const_iterator       it = ranks.find(i_glob);
        if (it == ranks.end()) return false;
        return it->second == get_own_rank();
    }
    int     get_total_size()const { return total_size; }
    int     get_size()const { return own_glob_indices.size(); }
    int     own_glob_ind(int i)const
    {
        assert((i >= 0)&&(i < get_size()));
        return own_glob_indices[i];
    }
    int     own_glob_ind_2_ind(int i_glob)const
    {
        std::map<int,int>::const_iterator       it = own_glob_indices_2_ind.find(i_glob);
        assert(it != own_glob_indices_2_ind.end());
        return it->second;
    }

    //'construction' part:
    void    add_stencil_element(int i_glob)
    {
        if (check_glob_owned(i_glob)) return;
        ranks[i_glob] = -1;
        if (use_linear_partitioner) linear_partitioner.add_stencil_element(i_glob);
    }
    void    complete()
    {
        is_complete = true;
        if (!use_linear_partitioner) {
            //TODO use c++ file streams (they are better for case of errors, because they close file in destructor)
            FILE    *f = fopen( f_name.c_str(), "rb" );
            if (f == NULL) throw std::runtime_error("color_file_partitioner::complete(): failed to open file " + f_name);
            if (fread(&total_size, sizeof(total_size), 1, f) != 1) throw std::runtime_error("color_file_partitioner::complete(): error while reading file " + f_name);
            int     file_colors_n;
            if (fread(&file_colors_n, sizeof(file_colors_n), 1, f) != 1) throw std::runtime_error("color_file_partitioner::complete(): error while reading file " + f_name);
            for (int i = 0;i < total_size;++i) {
                int color;
                if (fread(&color, sizeof(color), 1, f) != 1) throw std::runtime_error("color_file_partitioner::complete(): error while reading file " + f_name);
                std::map<int,int>::const_iterator it = ranks.find(i);
                if ((it != ranks.end())&&(color != my_rank))
                    ranks[i] = color;
            }
            fclose(f);
        } else {
            linear_partitioner.complete();
            for (std::map<int,int>::iterator it = ranks.begin();it != ranks.end();++it) {
                int     i_glob = it->first,
                    rank = it->second;
                if (rank != my_rank) it->second = linear_partitioner.get_rank(i_glob);
            }
        }
    }

    //'read' after construction part:
    //i_glob is ether index owner by calling process, ether index from stencil, otherwise behavoiur is  undefined
    //returns rank of process that owns i_glob index
    int     get_rank(int i_glob)const
    {
        assert(is_complete);
        std::map<int,int>::const_iterator it = ranks.find(i_glob);
        assert(it != ranks.end());
        return it->second;
    }
    //for (rank == get_own_rank()) result and behavoiur coincides with check_glob_owned(i_glob)
    //for rank != get_own_rank():
    //i_glob is ether index owner by calling process, ether index from stencil, otherwise behavoiur is  undefined
    //returns, if index i_glob is owned by rank processor
    bool    check_glob_owned(int i_glob, int rank)const 
    { 
        assert(is_complete);
        if (rank == get_own_rank()) {
            return check_glob_owned(i_glob);
        } else {
            return get_rank(i_glob) == rank; 
        }
    }
};

}

#endif
