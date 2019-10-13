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

#ifndef __SCFD_BLOCK_VECTOR_OPERATIONS_IMPL_H__
#define __SCFD_BLOCK_VECTOR_OPERATIONS_IMPL_H__

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <utils/cuda_safe_call.h>
#include <tensor_field/t_tensor_field_tml.h>
#include <for_each/for_each_1d.h>
#include <for_each/for_each_1d_cuda_impl.cuh>
#include <for_each/for_each_func_macro.h>
#include "block_vector.h"
#include "block_vector_operations.h"

//TODO only CUDA variant is supported for now
//TODO kernels organisation 1 thread per 1 element instead of block

namespace numerical_algos
{

template<class T>
struct block_vector_check_is_valid_number_func
{
    FOR_EACH_FUNC_PARAMS_HELP(block_vector_check_is_valid_number_func, 
                              const T*, x, T*, flag)

    __DEVICE_TAG__ void operator()(const int &i)const 
    {
        flag[i] = (isnan(x[i])?-T(1.f):T(1.f));
    }
};

template<class T>
struct block_vector_scalar_prod_func
{
    FOR_EACH_FUNC_PARAMS_HELP(block_vector_scalar_prod_func, 
                              const T*, w, const T*, x, const T*, y, T*, res)

    __DEVICE_TAG__ void operator()(const int &i)const 
    {
        res[i] = w[i]*x[i]*y[i];
    }
};

//v := <vector with scalar value>
template<class T>
struct block_vector_assign_scalar_func
{
    FOR_EACH_FUNC_PARAMS_HELP(block_vector_assign_scalar_func, 
                              T, scalar, T*, x)

    __DEVICE_TAG__ void operator()(const int &i)const 
    {
        x[i] = scalar;
    }
};

template<class T>
struct block_vector_add_mul_scalar_func
{
    FOR_EACH_FUNC_PARAMS_HELP(block_vector_add_mul_scalar_func, 
                              T, scalar, T, mul_x, T*, x)

    __DEVICE_TAG__ void operator()(const int &i)const 
    {
        x[i] = mul_x*x[i] + scalar;
    }
};

//y := x
template<class T>
struct block_vector_assign_func
{
    FOR_EACH_FUNC_PARAMS_HELP(block_vector_assign_func, 
                              const T*, x, T*, y)

    __DEVICE_TAG__ void operator()(const int &i)const 
    {
        y[i] = x[i];
    }
};

//y := mul_x*x
template<class T>
struct block_vector_assign_mul_1_func
{
    FOR_EACH_FUNC_PARAMS_HELP(block_vector_assign_mul_1_func, 
                              T, mul_x, const T*, x, T*, y)

    __DEVICE_TAG__ void operator()(const int &i)const 
    {
        y[i] = mul_x*x[i];
    }
};

//z := mul_x*x + mul_y*y
template<class T>
struct block_vector_assign_mul_2_func
{
    FOR_EACH_FUNC_PARAMS_HELP(block_vector_assign_mul_2_func, 
                              T, mul_x, const T*, x, T, mul_y, const T*, y, T*, z)

    __DEVICE_TAG__ void operator()(const int &i)const 
    {
        z[i] = mul_x*x[i] + mul_y*y[i];
    }
};

//y := mul_x*x + mul_y*y
template<class T>
struct block_vector_add_mul_1_func
{
    FOR_EACH_FUNC_PARAMS_HELP(block_vector_add_mul_1_func, 
                              T, mul_x, const T*, x, T, mul_y, T*, y)

    __DEVICE_TAG__ void operator()(const int &i)const 
    {
        y[i] = mul_x*x[i] + mul_y*y[i];
    }
};

//z := mul_x*x + mul_y*y + mul_z*z
template<class T>
struct block_vector_add_mul_2_func
{
    FOR_EACH_FUNC_PARAMS_HELP(block_vector_add_mul_2_func, 
                              T, mul_x, const T*, x, T, mul_y, const T*, y, T, mul_z, T*, z)

    __DEVICE_TAG__ void operator()(const int &i)const 
    {
        z[i] = mul_x*x[i] + mul_y*y[i] + mul_z*z[i];
    }
};

template<class T>
void mpi_allreduce_(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
{
    throw std::logic_error("mpi_allreduce_: unknown type");
}

template<>
void mpi_allreduce_<float>(float *sendbuf, float *recvbuf, int count, MPI_Op op, MPI_Comm comm)
{
    MPI_Allreduce( (void*)sendbuf, (void *)recvbuf, count, MPI_FLOAT, op, comm );
}

template<>
void mpi_allreduce_<double>(double *sendbuf, double *recvbuf, int count, MPI_Op op, MPI_Comm comm)
{
    MPI_Allreduce( (void*)sendbuf, (void *)recvbuf, count, MPI_DOUBLE, op, comm );
}

template<class T,t_tensor_field_storage storage,class Map,class WeightOperator>
bool            block_vector_operations<T,storage,Map,WeightOperator>::check_is_valid_number(const vector_type &x)const
{
    for_each_( block_vector_check_is_valid_number_func<T>( x.ptr(), buf_.ptr() ),
               map_->min_own_loc_ind()*block_size_, (map_->max_own_loc_ind()+1)*block_size_ );
    thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(buf_.ptr());
    T   _cons = thrust::reduce(dev_ptr + map_->min_own_loc_ind()*block_size_, 
                               dev_ptr + (map_->max_own_loc_ind()+1)*block_size_, 
                               T(1.f), thrust::minimum<T>()),
        cons = _cons;
#ifdef SCFD_BLOCK_VECTOR_ENABLE_MPI
    mpi_allreduce_( &_cons, &cons, 1, MPI_MIN, MPI_COMM_WORLD );
#endif 
    return cons > T(0.f);    
}

template<class T,t_tensor_field_storage storage,class Map,class WeightOperator>
T     block_vector_operations<T,storage,Map,WeightOperator>::norm(const vector_type &x)const
{
    if (op_ != NULL) {
        op_->apply(x, buf2_);
        return sqrt(scalar_prod(buf2_,buf2_));
    } else
        return sqrt(scalar_prod(x,x));
}

template<class T,t_tensor_field_storage storage,class Map,class WeightOperator>
T     block_vector_operations<T,storage,Map,WeightOperator>::scalar_prod(const vector_type &x, const vector_type &y)const
{
    for_each_( block_vector_scalar_prod_func<T>( w_.ptr(), x.ptr(), y.ptr(), buf_.ptr() ),
               map_->min_own_loc_ind()*block_size_, (map_->max_own_loc_ind()+1)*block_size_ );
    thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(buf_.ptr());
    T _res = thrust::reduce(dev_ptr + map_->min_own_loc_ind()*block_size_, 
                            dev_ptr + (map_->max_own_loc_ind()+1)*block_size_, 
                            T(0.f), thrust::plus<T>());
    T __res = _res;
#ifdef SCFD_BLOCK_VECTOR_ENABLE_MPI
    mpi_allreduce_( &_res, &__res, 1, MPI_SUM, MPI_COMM_WORLD );
#endif
    return __res;
}

/*template<class T,t_tensor_field_storage storage,class Map>
void            block_vector_operations<T,storage,Map>::assign_nan(vector_type& x)const
{
    for_each_( block_vector_assign_scalar_func<T>( T(0.)/T(0.), x.ptr() ),
               map_->min_loc_ind()*block_size_, (map_->max_loc_ind()+1)*block_size_ );
}*/

//calc: x := <vector_type with all elements equal to given scalar value> 
template<class T,t_tensor_field_storage storage,class Map,class WeightOperator>
void            block_vector_operations<T,storage,Map,WeightOperator>::assign_scalar(scalar_type scalar, vector_type& x)const
{
    for_each_( block_vector_assign_scalar_func<T>( scalar, x.ptr() ),
               map_->min_own_loc_ind()*block_size_, (map_->max_own_loc_ind()+1)*block_size_ );
}
//calc: x := mul_x*x + <vector_type of all scalar value> 
template<class T,t_tensor_field_storage storage,class Map,class WeightOperator>
void            block_vector_operations<T,storage,Map,WeightOperator>::add_mul_scalar(scalar_type scalar, scalar_type mul_x, vector_type& x)const
{
    for_each_( block_vector_add_mul_scalar_func<T>( scalar, mul_x, x.ptr() ),
               map_->min_own_loc_ind()*block_size_, (map_->max_own_loc_ind()+1)*block_size_ );
}
//copy: y := x
template<class T,t_tensor_field_storage storage,class Map,class WeightOperator>
void            block_vector_operations<T,storage,Map,WeightOperator>::assign(const vector_type& x, vector_type& y)const
{
    for_each_( block_vector_assign_func<T>( x.ptr(), y.ptr() ),
               map_->min_own_loc_ind()*block_size_, (map_->max_own_loc_ind()+1)*block_size_ );
}
//calc: y := mul_x*x
template<class T,t_tensor_field_storage storage,class Map,class WeightOperator>
void            block_vector_operations<T,storage,Map,WeightOperator>::assign_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
{
    for_each_( block_vector_assign_mul_1_func<T>( mul_x, x.ptr(), y.ptr() ),
               map_->min_own_loc_ind()*block_size_, (map_->max_own_loc_ind()+1)*block_size_ );
}
//calc: z := mul_x*x + mul_y*y
template<class T,t_tensor_field_storage storage,class Map,class WeightOperator>
void            block_vector_operations<T,storage,Map,WeightOperator>::assign_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, 
                                                                   vector_type& z)const
{
    for_each_( block_vector_assign_mul_2_func<T>( mul_x, x.ptr(), mul_y, y.ptr(), z.ptr() ),
               map_->min_own_loc_ind()*block_size_, (map_->max_own_loc_ind()+1)*block_size_ );
}
//calc: y := mul_x*x + mul_y*y
template<class T,t_tensor_field_storage storage,class Map,class WeightOperator>
void            block_vector_operations<T,storage,Map,WeightOperator>::add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, vector_type& y)const
{
    for_each_( block_vector_add_mul_1_func<T>( mul_x, x.ptr(), mul_y, y.ptr() ),
               map_->min_own_loc_ind()*block_size_, (map_->max_own_loc_ind()+1)*block_size_ );
}
//calc: z := mul_x*x + mul_y*y + mul_z*z
template<class T,t_tensor_field_storage storage,class Map,class WeightOperator>
void            block_vector_operations<T,storage,Map,WeightOperator>::add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, 
                                                                scalar_type mul_z, vector_type& z)const
{
    for_each_( block_vector_add_mul_2_func<T>( mul_x, x.ptr(), mul_y, y.ptr(), mul_z, z.ptr() ),
               map_->min_own_loc_ind()*block_size_, (map_->max_own_loc_ind()+1)*block_size_ );
}

}

#endif
