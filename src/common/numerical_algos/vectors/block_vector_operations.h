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

#ifndef __SCFD_BLOCK_VECTOR_OPERATIONS_H__
#define __SCFD_BLOCK_VECTOR_OPERATIONS_H__

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <utils/cuda_safe_call.h>
#include <tensor_field/t_tensor_field_tml.h>
#include <for_each/for_each_1d.h>
#include "block_vector.h"

//TODO only CUDA variant is supported for now
//TODO kernels organisation 1 thread per 1 element instead of block

namespace numerical_algos
{

template<class Vector>
struct EmptyOperator
{
    void apply(const Vector &x, Vector &y)
    {
    }
};

template<class T,t_tensor_field_storage storage,class Map, 
         class WeightOperator = EmptyOperator<block_vector<T,storage,Map> > >
class block_vector_operations
{
public:
    typedef T                               scalar_type;
    typedef block_vector<T,storage,Map>     vector_type;
    typedef WeightOperator                  weight_operator_type;

private:
    static const t_for_each_type            for_each_type = FET_CUDA;
    typedef for_each_1d<for_each_type>      for_each_t;

    const Map               *map_;
    for_each_t              for_each_;
    int                     block_size_;
    mutable vector_type     buf_, buf2_;
    vector_type             w_;
    weight_operator_type    *op_;

    //we simply forbit this operations for now
    block_vector_operations(const block_vector_operations &vo) { } 
    block_vector_operations &operator=(const block_vector_operations &vo) { return *this; }
public:
    block_vector_operations() : map_(NULL), op_(NULL) { }
    block_vector_operations(const Map *map, int block_size) : map_(map), block_size_(block_size), op_(NULL)
    {
        init(map, block_size);
    }

    void            init(const Map *map, int block_size)
    {
        map_ = map; block_size_ = block_size;
        buf_.init(*map_, block_size_); w_.init(*map_, block_size_);  
        buf2_.init(*map_, block_size_); 
    }

    void            set_scalar_prod_weights(vector_type& w)
    {
        assign(w, w_);
    }
    void            set_weight_operator(weight_operator_type  *op) { op_ = op; }

    void            init_vector(vector_type& x)const 
    {
    }
    void            free_vector(vector_type& x)const 
    {
    }
    void            start_use_vector(vector_type& x)const
    {
        //TODO check sizes coincidence (what to do? reinit? throw an error?)
        if (!x.is_inited()) x.init(*map_, block_size_);
    }
    void            stop_use_vector(vector_type& x)const
    {
    }

    bool            check_is_valid_number(const vector_type &x)const;
    scalar_type     norm(const vector_type &x)const;
    scalar_type     scalar_prod(const vector_type &x, const vector_type &y)const;    
    //void            assign_nan(vector_type& x)const;
    //calc: x := <vector_type with all elements equal to given scalar value> 
    void            assign_scalar(scalar_type scalar, vector_type& x)const;
    //calc: x := mul_x*x + <vector_type of all scalar value> 
    void            add_mul_scalar(scalar_type scalar, scalar_type mul_x, vector_type& x)const;
    //copy: y := x
    void            assign(const vector_type& x, vector_type& y)const;
    //calc: y := mul_x*x
    void            assign_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const;
    //calc: z := mul_x*x + mul_y*y
    void            assign_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, 
                               vector_type& z)const;
    //calc: y := mul_x*x + mul_y*y
    void            add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, vector_type& y)const;
    //calc: z := mul_x*x + mul_y*y + mul_z*z
    void            add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, 
                            scalar_type mul_z, vector_type& z)const;
};

}

#endif
