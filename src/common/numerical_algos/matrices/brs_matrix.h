// Copyright © 2016-2018 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_BRS_MATRIX_H__
#define __SCFD_BRS_MATRIX_H__

#include <cassert>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <utils/cuda_safe_call.h>
#include <utils/cusparse_safe_call.h>
#include <utils/device_tag.h>
#include <numerical_algos/vectors/block_vector.h>
#include <communication/mpi_distributor.h>
//TODO move this include somewhere
#include <communication/mpi_distributor_copy_kernels_impl.h>
#include <for_each/for_each_1d.h>
#include <tensor_field/t_tensor_field_tml.h>
#include "brs_matrix_structure.h"

#define BRS_MATRIX_MAX_BLOCKDIM 5
//TODO temporal solution
#define BRS_MATRIX_CUSPARSE_DIR_TYPE       CUSPARSE_DIRECTION_COLUMN
#define BRS_MATRIX_BLK_IND(i1,i2)          ((i2)*block_size+(i1))

namespace numerical_algos
{

//ISSUE what about block sizes? are they static or dynamic?

template<class T,t_tensor_field_storage storage,class Map>
class brs_matrix
{
public:
    typedef brs_matrix_structure<T,storage,Map>             structure_type;
    typedef Map                                             map_type;
    typedef block_vector<T,storage,Map>                     block_vector_type;
private:
    static const t_for_each_type                            for_each_type = FET_CUDA;
    typedef for_each_1d<for_each_type>                      for_each_t;

    structure_type                      *mat_str_;
    t_tensor0_field_tml<T,storage>       vals_;

    mutable block_vector_type            tmp1_, tmp2_; 

    cusparseHandle_t        handle_;
    mutable void            *p_buf_l_, *p_buf_r_; 
    mutable bsrsv2Info_t    info_l_, info_r_;
    for_each_t              for_each_;

    brs_matrix(const brs_matrix &mat) { }
    brs_matrix &operator=(const brs_matrix &mat) { return *this; }
public:
    brs_matrix() {}

    void    init(cusparseHandle_t handle, structure_type *matrix_structure);
    //read_from_file called for already inited matrix
    void    read_from_file(const std::string &fn);
    void    free()
    {
    }

    const t_tensor0_field_tml<T,storage>    &vals() { return vals_; }

    void apply(const block_vector_type &x, block_vector_type &y)const;
    void apply_inverted_upper(const block_vector_type &x, block_vector_type &y)const;
    void apply_inverted_lower(const block_vector_type &x, block_vector_type &y)const;

    void copy_diagonal(brs_matrix &diag_mat)const;
    void copy_inverted_diagonal(brs_matrix &diag_mat)const;
    void left_mul_by_diagonal_matrix(const brs_matrix &diag_mat);
    void add_mul_ident_matrix(T alpha, T beta);

    ~brs_matrix()
    {
        free();
    }
};

}

#endif
