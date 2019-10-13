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

#ifndef __SCFD_BRS_MATRIX_IMPL_H__
#define __SCFD_BRS_MATRIX_IMPL_H__

#include <cassert>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <utils/cuda_safe_call.h>
#include <utils/cusparse_safe_call.h>
#include <utils/device_tag.h>
#include <numerical_algos/vectors/block_vector.h>
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
#include <communication/mpi_distributor.h>
//TODO move this include somewhere
#include <communication/mpi_distributor_copy_kernels_impl.h>
#endif
#include <for_each/for_each_1d.h>
#include <for_each/for_each_1d_cuda_impl.cuh>
#include <for_each/for_each_func_macro.h>
#include <tensor_field/t_tensor_field_tml.h>
#include "brs_matrix_structure.h"
#include "brs_matrix_impl.h"

namespace numerical_algos
{

template<class T>
cusparseStatus_t cusparseXbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const cusparseMatDescr_t descr, T *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes)
{
}

template<>
cusparseStatus_t cusparseXbsrsv2_bufferSize<float>(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const cusparseMatDescr_t descr, float *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes)
{
    return  cusparseSbsrsv2_bufferSize(handle, dir, trans, mb, nnzb, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes);
}

template<>
cusparseStatus_t cusparseXbsrsv2_bufferSize<double>(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const cusparseMatDescr_t descr, double *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes)
{
    return  cusparseDbsrsv2_bufferSize(handle, dir, trans, mb, nnzb, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes);
}

template<class T>
cusparseStatus_t cusparseXbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const cusparseMatDescr_t descr, const T *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
{
}

template<>
cusparseStatus_t cusparseXbsrsv2_analysis<float>(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const cusparseMatDescr_t descr, const float *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
{
    return cusparseSbsrsv2_analysis(handle, dir, trans, mb, nnzb, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
}

template<>
cusparseStatus_t cusparseXbsrsv2_analysis<double>(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const cusparseMatDescr_t descr, const double *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
{
    return cusparseDbsrsv2_analysis(handle, dir, trans, mb, nnzb, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
}

template<class T,t_tensor_field_storage storage,class Map>
void brs_matrix<T,storage,Map>::init(cusparseHandle_t handle, structure_type *matrix_structure)
{
    handle_ = handle;
    mat_str_ = matrix_structure;
    vals_.init(mat_str_->loc_nonzeros_n_*mat_str_->block_row_size_*mat_str_->block_col_size_);
    tmp1_.init(mat_str_->block_row_size_, mat_str_->loc_cols_n_);
    tmp2_.init(mat_str_->block_row_size_, mat_str_->loc_cols_n_);

    cusparseMatDescr_t      descr_l = 0, descr_r = 0;

    CUSPARSE_SAFE_CALL( cusparseCreateMatDescr(&descr_l) );
    cusparseSetMatType(descr_l,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_l,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatDiagType(descr_l, CUSPARSE_DIAG_TYPE_UNIT);
    cusparseSetMatFillMode(descr_l, CUSPARSE_FILL_MODE_LOWER);

    CUSPARSE_SAFE_CALL( cusparseCreateMatDescr(&descr_r) );
    cusparseSetMatType(descr_r,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_r,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatDiagType(descr_r, CUSPARSE_DIAG_TYPE_UNIT);
    cusparseSetMatFillMode(descr_r, CUSPARSE_FILL_MODE_UPPER);

    cusparseSolvePolicy_t   policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    cusparseDirection_t     dir = BRS_MATRIX_CUSPARSE_DIR_TYPE; 
    cusparseOperation_t     trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

    info_l_ = 0; p_buf_l_ = 0;
    int p_buf_l_size;

    CUSPARSE_SAFE_CALL( cusparseCreateBsrsv2Info(&info_l_) );

    CUSPARSE_SAFE_CALL( cusparseXbsrsv2_bufferSize(handle_, dir, trans, mat_str_->loc_rows_n_, mat_str_->loc_nonzeros_n_, 
                                                   descr_l, vals_.d, mat_str_->row_ptr_.d, mat_str_->col_ind_no_borders_.d, 
                                                   mat_str_->block_row_size_, info_l_, &p_buf_l_size) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&p_buf_l_, p_buf_l_size) );

    //ANALYSIS
    CUSPARSE_SAFE_CALL( cusparseXbsrsv2_analysis(handle_, dir, trans, mat_str_->loc_rows_n_, mat_str_->loc_nonzeros_n_, 
                                                 descr_l, vals_.d, mat_str_->row_ptr_.d, mat_str_->col_ind_no_borders_.d, 
                                                 mat_str_->block_row_size_, info_l_, policy, p_buf_l_) );

    info_r_ = 0; p_buf_r_ = 0;
    int p_buf_r_size;

    CUSPARSE_SAFE_CALL( cusparseCreateBsrsv2Info(&info_r_) );

    CUSPARSE_SAFE_CALL( cusparseXbsrsv2_bufferSize(handle_, dir, trans, mat_str_->loc_rows_n_, mat_str_->loc_nonzeros_n_, 
                                                   descr_r, vals_.d, mat_str_->row_ptr_.d, mat_str_->col_ind_no_borders_.d, 
                                                   mat_str_->block_row_size_, info_r_, &p_buf_r_size) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&p_buf_r_, p_buf_r_size) );

    //ANALYSIS
    CUSPARSE_SAFE_CALL( cusparseXbsrsv2_analysis(handle_, dir, trans, mat_str_->loc_rows_n_, mat_str_->loc_nonzeros_n_, 
                                                 descr_r, vals_.d, mat_str_->row_ptr_.d, mat_str_->col_ind_no_borders_.d, 
                                                 mat_str_->block_row_size_, info_r_, policy, p_buf_r_) );
}

template<class T,t_tensor_field_storage storage,class Map>
void brs_matrix<T,storage,Map>::read_from_file(const std::string &fn)
{
    std::ifstream f(fn.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("brs_matrix::read_from_file: error while opening file " + fn);

    std::string     buf;
    int             algebraic_rows_n, algebraic_cols_n, 
                    algebraic_nonzeros_n;
    int             block_row_size_, block_col_size_;
    int             glob_rows_n_, glob_cols_n_;
    int             glob_nonzeros_n_;

    if (!getline(f,buf)) throw std::runtime_error("brs_matrix::read_from_file: error while reading first line");
    if (!(f >> buf >> block_row_size_ >> block_col_size_)) throw std::runtime_error("brs_matrix::read_from_file: error while scanf");
    if (block_row_size_ != mat_str_->block_row_size_) throw std::runtime_error("brs_matrix::read_from_file: row block size is not equal to one in matrix structure");
    if (block_col_size_ != mat_str_->block_col_size_) throw std::runtime_error("brs_matrix::read_from_file: col block size is not equal to one in matrix structure");
    if (!(f >> algebraic_rows_n >> algebraic_cols_n >> algebraic_nonzeros_n)) throw std::runtime_error("brs_matrix::read_from_file: error while read header");
    if (algebraic_rows_n%block_row_size_ != 0) throw std::runtime_error("brs_matrix::read_from_file: matrix size is not divider of block size");
    if (algebraic_cols_n%block_col_size_ != 0) throw std::runtime_error("brs_matrix::read_from_file: matrix size is not divider of block size");
    if (algebraic_nonzeros_n%(block_row_size_*block_col_size_) != 0) throw std::runtime_error("brs_matrix::read_from_file: matrix nonzero size is not divider of block size square");
    glob_rows_n_ = algebraic_rows_n/block_row_size_;
    glob_cols_n_ = algebraic_cols_n/block_col_size_;
    glob_nonzeros_n_ = algebraic_nonzeros_n/(block_row_size_*block_col_size_);
    if (glob_rows_n_ != mat_str_->glob_rows_n_) throw std::runtime_error("brs_matrix::read_from_file: rows size is not equal to one in matrix structure");
    if (glob_cols_n_ != mat_str_->glob_cols_n_) throw std::runtime_error("brs_matrix::read_from_file: cols size is not equal to one in matrix structure");

    //TODO temporal solution
    int block_size = block_row_size_;

    t_tensor0_field_view_tml<T,storage>     vals_view(vals_, false);
    for (int i = 0;i < glob_nonzeros_n_;++i) {
        int     col, row;
        bool    is_own;
        int     elem_ptr;
        for (int ii1_ = 0;ii1_ < block_row_size_;++ii1_)
        for (int ii2_ = 0;ii2_ < block_col_size_;++ii2_) {
            int     col_, row_;
            T       val;
            if (!(f >> row_ >> col_ >> val)) throw std::runtime_error("brs_matrix::read_from_file: error while reading vals");
            row_ = row_/block_row_size_;
            col_ = col_/block_col_size_;
            int     ii1 = row_%block_row_size_,
                    ii2 = col_%block_col_size_;
            if ((ii1_ == 0)&&(ii2_ == 0)) {
                col = col_; row = row_;
                is_own = mat_str_->map_->check_glob_owned(row);
                if (is_own) elem_ptr = mat_str_->find_elem_ptr(row, col);
            } else {
                if ((col != col_)||(row != row_)) throw std::runtime_error("brs_matrix::read_from_file: blocks are intermitted");
            }
            if (is_own) vals_view(elem_ptr*block_size*block_size+BRS_MATRIX_BLK_IND(ii1,ii2)) = val;
        }
    }
    vals_view.release(true);
}

template<class T>
cusparseStatus_t cusparseXbsrmv(cusparseHandle_t handle_, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nb, int nnzb, const T *alpha, const cusparseMatDescr_t descr, const T *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, const T *x, const T *beta, T *y)
{
}

template<>
cusparseStatus_t cusparseXbsrmv<float>(cusparseHandle_t handle_, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nb, int nnzb, const float *alpha, const cusparseMatDescr_t descr, const float *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, const float *x, const float *beta, float *y)
{
    return cusparseSbsrmv(handle_, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y);
}

template<>
cusparseStatus_t cusparseXbsrmv<double>(cusparseHandle_t handle_, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nb, int nnzb, const double *alpha, const cusparseMatDescr_t descr, const double *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, const double *x, const double *beta, double *y)
{
    return cusparseDbsrmv(handle_, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y);
}

template<class T,t_tensor_field_storage storage,class Map>
void brs_matrix<T,storage,Map>::apply(const block_vector_type &x, block_vector_type &y)const
{
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
    mat_str_->distributor_.set_data(x.ptr(), x.size(), x.block_size(), false);
    mat_str_->distributor_.sync();
#endif

    cusparseDirection_t     dir = BRS_MATRIX_CUSPARSE_DIR_TYPE; 
    cusparseOperation_t     trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
    T                       one(1.f);
    T                       zero(0.f);

    cusparseMatDescr_t      descr = 0; 

    CUSPARSE_SAFE_CALL( cusparseCreateMatDescr(&descr) );
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);

    //TODO how about fill rhs with zeros or something?
    //CUDA_SAFE_CALL( cudaMemcpy( x_new.d, rhs.d, rhs.N*rhs.block_sz*sizeof(T), cudaMemcpyDeviceToDevice ) );

    //TODO check whether x.block_size(), f.block_size(), block_row_size_ 
    //and block_col_size_ are the same

    //Matrix multiplication
    CUSPARSE_SAFE_CALL( cusparseXbsrmv<T>(handle_, dir, trans, 
                                          mat_str_->loc_rows_n_, mat_str_->loc_cols_n_, mat_str_->loc_nonzeros_n_, 
                                          &one, descr, vals_.d, mat_str_->row_ptr_.d, mat_str_->col_ind_.d, 
                                          x.block_size(), x.ptr(), &zero, y.ptr()) );
}

template<class T>
struct brs_matrix_update_colored_func
{
    FOR_EACH_FUNC_PARAMS_HELP(brs_matrix_update_colored_func,
                              int, curr_color, bool, do_upper, int*, colors, 
                              int*, row_ptrs, int*, col_inds, T*, vals,
                              int, block_size, const T*, rhs, T*, x)

    __DEVICE_TAG__ void operator()(const int &row)const
    {
        int color = colors[row];
        if (color != curr_color) return;

        int row_begin = row_ptrs[row],
            row_end = row_ptrs[row+1];
        T   res_block[BRS_MATRIX_MAX_BLOCKDIM];
        #pragma unroll 
        for (int ii = 0;ii < BRS_MATRIX_MAX_BLOCKDIM;++ii) {
            if (!(ii < block_size)) break;
            res_block[ii] = rhs[ row*block_size + ii ];
        }
        for (int row_ptr = row_begin;row_ptr < row_end;++row_ptr) {
            int col = col_inds[row_ptr];
            if (col == row) continue;
            int color = colors[col];
            if ( do_upper && (color < curr_color)) continue;
            if (!do_upper && (color > curr_color)) continue;
            //NOTE color == curr_color here is logical error
            T   x_block[BRS_MATRIX_MAX_BLOCKDIM];
            #pragma unroll 
            for (int ii = 0;ii < BRS_MATRIX_MAX_BLOCKDIM;++ii) {
                if (!(ii < block_size)) break;
                x_block[ii] = x[ col*block_size + ii ];
            }
            #pragma unroll
            for (int ii1 = 0;ii1 < BRS_MATRIX_MAX_BLOCKDIM;++ii1) {
                if (!(ii1 < block_size)) break;
                #pragma unroll
                for (int ii2 = 0;ii2 < BRS_MATRIX_MAX_BLOCKDIM;++ii2) {
                    if (!(ii2 < block_size)) break;
                    res_block[ii1] -= vals[row_ptr*block_size*block_size+BRS_MATRIX_BLK_IND(ii1,ii2)] * x_block[ii2];
                }
            }
        }
        #pragma unroll 
        for (int ii = 0;ii < BRS_MATRIX_MAX_BLOCKDIM;++ii) {
            if (!(ii < block_size)) break;
            x[ row*block_size + ii ] = res_block[ii];
        }
    }

};

template<class T>
struct brs_matrix_prepare_fake_rhs_func
{
    FOR_EACH_FUNC_PARAMS_HELP(brs_matrix_prepare_fake_rhs_func,
                              bool, do_upper, int*, colors, 
                              int*, row_ptrs, int*, col_inds, T*, vals,
                              int, block_size, const T*, real_rhs, const T*, x, T*, fake_rhs)

    __DEVICE_TAG__ void operator()(const int &row)const
    {
        int color = colors[row];
        if (color != 0) {
            #pragma unroll 
            for (int ii = 0;ii < BRS_MATRIX_MAX_BLOCKDIM;++ii) {
                if (!(ii < block_size)) break;
                fake_rhs[ row*block_size + ii ] = T(0.f);
            }
            return;
        }

        int row_begin = row_ptrs[row],
            row_end = row_ptrs[row+1];
        T   res_block[BRS_MATRIX_MAX_BLOCKDIM];
        #pragma unroll 
        for (int ii = 0;ii < BRS_MATRIX_MAX_BLOCKDIM;++ii) {
            if (!(ii < block_size)) break;
            res_block[ ii ] = real_rhs[ row*block_size + ii ];
        }

        if (do_upper) {
            for (int row_ptr = row_begin;row_ptr < row_end;++row_ptr) {
                int col = col_inds[row_ptr];
                if (col == row) continue;
                int color = colors[col];
                if (color == 0) continue;
                T   x_block[BRS_MATRIX_MAX_BLOCKDIM];
                #pragma unroll 
                for (int ii = 0;ii < BRS_MATRIX_MAX_BLOCKDIM;++ii) {
                    if (!(ii < block_size)) break;
                    x_block[ii] = x[ col*block_size + ii ];
                }
                #pragma unroll
                for (int ii1 = 0;ii1 < BRS_MATRIX_MAX_BLOCKDIM;++ii1) {
                    if (!(ii1 < block_size)) break;
                    #pragma unroll
                    for (int ii2 = 0;ii2 < BRS_MATRIX_MAX_BLOCKDIM;++ii2) {
                        if (!(ii2 < block_size)) break;
                        res_block[ii1] -= vals[row_ptr*block_size*block_size+BRS_MATRIX_BLK_IND(ii1,ii2)] * x_block[ii2];
                    }
                }
            }
        }

        #pragma unroll 
        for (int ii = 0;ii < BRS_MATRIX_MAX_BLOCKDIM;++ii) {
            if (!(ii < block_size)) break;
            fake_rhs[ row*block_size + ii ] = res_block[ ii ];
        }        
    }

};

template<class T>
struct brs_matrix_copy_colored_res_func
{
    FOR_EACH_FUNC_PARAMS_HELP(brs_matrix_copy_colored_res_func,
                              int*, colors, int, block_size, T*, colored_elems_res, T*, final_res)

    __DEVICE_TAG__ void operator()(const int &row)const
    {
        int color = colors[row];
        if (color == 0) return;

        #pragma unroll 
        for (int ii = 0;ii < BRS_MATRIX_MAX_BLOCKDIM;++ii) {
            if (!(ii < block_size)) break;
            final_res[ row*block_size + ii ] = colored_elems_res[ row*block_size + ii ];
        }
    }

};

template<class T>
cusparseStatus_t cusparseXbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const T *alpha, const cusparseMatDescr_t descr, const T *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, bsrsv2Info_t info, const T *x, T *y, cusparseSolvePolicy_t policy, void *pBuffer)
{
}

template<>
cusparseStatus_t cusparseXbsrsv2_solve<float>(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const float *alpha, const cusparseMatDescr_t descr, const float *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, bsrsv2Info_t info, const float *x, float *y, cusparseSolvePolicy_t policy, void *pBuffer)
{
    return cusparseSbsrsv2_solve(handle, dir, trans, mb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, x, y, policy, pBuffer);
}

template<>
cusparseStatus_t cusparseXbsrsv2_solve<double>(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const double *alpha, const cusparseMatDescr_t descr, const double *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, bsrsv2Info_t info, const double *x, double *y, cusparseSolvePolicy_t policy, void *pBuffer)
{
    return cusparseDbsrsv2_solve(handle, dir, trans, mb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, x, y, policy, pBuffer);
}

//NOTE x here is RHS for the system
template<class T,t_tensor_field_storage storage,class Map>
void brs_matrix<T,storage,Map>::apply_inverted_upper(const block_vector_type &x, block_vector_type &y)const
{
    //calc solution for colors>0 - border elements

    for (int color = mat_str_->colors_n_-1; color >= 1;--color) {
        for_each_( brs_matrix_update_colored_func<T>( color, true, mat_str_->colors_.d, 
                                                      mat_str_->row_ptr_.d, mat_str_->col_ind_.d, vals_.d,
                                                      mat_str_->block_row_size_, x.ptr(), tmp1_.ptr() ),
                  0, mat_str_->loc_rows_n_ );

        if (color >= 2) {
//TODO maybe cover all this block?
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
            mat_str_->colors_distributors_[color-1].set_data(tmp1_.ptr(), tmp1_.size(), tmp1_.block_size(), false);
            mat_str_->colors_distributors_[color-1].sync();
#endif
        }
    }

    //calc color zero - internal elements

    //prepare proper rhs for trinagular solver
    for_each_( brs_matrix_prepare_fake_rhs_func<T>( true, mat_str_->colors_.d, 
                                                    mat_str_->row_ptr_.d, mat_str_->col_ind_.d, vals_.d,
                                                    mat_str_->block_row_size_, x.ptr(), tmp1_.ptr(), tmp2_.ptr()),
              0, mat_str_->loc_rows_n_ );

    cusparseSolvePolicy_t   policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    cusparseDirection_t     dir = BRS_MATRIX_CUSPARSE_DIR_TYPE; 
    cusparseOperation_t     trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
    T                       one(1.f);

    cusparseMatDescr_t      descr_u = 0;

    CUSPARSE_SAFE_CALL( cusparseCreateMatDescr(&descr_u) );
    cusparseSetMatType(descr_u,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_u,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatDiagType(descr_u, CUSPARSE_DIAG_TYPE_UNIT);
    cusparseSetMatFillMode(descr_u, CUSPARSE_FILL_MODE_UPPER);

    CUSPARSE_SAFE_CALL( cusparseXbsrsv2_solve(handle_, dir, trans, mat_str_->loc_rows_n_, mat_str_->loc_nonzeros_n_, 
                                              &one, descr_u, vals_.d, mat_str_->row_ptr_.d, mat_str_->col_ind_no_borders_.d, 
                                              mat_str_->block_row_size_, info_r_, tmp2_.ptr(), y.ptr(), policy, p_buf_r_) );

    //copy colored elements solution from tmp1_ to y
    for_each_( brs_matrix_copy_colored_res_func<T>( mat_str_->colors_.d, mat_str_->block_row_size_, tmp1_.ptr(), y.ptr()),
               0, mat_str_->loc_rows_n_ );
}

//NOTE x here is RHS for the system
template<class T,t_tensor_field_storage storage,class Map>
void brs_matrix<T,storage,Map>::apply_inverted_lower(const block_vector_type &x, block_vector_type &y)const
{
    //calc color zero - internal elements

    //prepare proper rhs for trinagular solver
    for_each_( brs_matrix_prepare_fake_rhs_func<T>( false, mat_str_->colors_.d, 
                                                    mat_str_->row_ptr_.d, mat_str_->col_ind_.d, vals_.d,
                                                    mat_str_->block_row_size_, x.ptr(), y.ptr(), tmp1_.ptr()),
              0, mat_str_->loc_rows_n_ );

    cusparseSolvePolicy_t   policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    cusparseDirection_t     dir = BRS_MATRIX_CUSPARSE_DIR_TYPE; 
    cusparseOperation_t     trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
    T                       one(1.f);

    cusparseMatDescr_t      descr_l = 0; 

    CUSPARSE_SAFE_CALL( cusparseCreateMatDescr(&descr_l) );
    cusparseSetMatType(descr_l,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_l,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatDiagType(descr_l, CUSPARSE_DIAG_TYPE_UNIT);
    cusparseSetMatFillMode(descr_l, CUSPARSE_FILL_MODE_LOWER);

    //triangular matrix inversion    
    CUSPARSE_SAFE_CALL( cusparseXbsrsv2_solve(handle_, dir, trans, mat_str_->loc_rows_n_, mat_str_->loc_nonzeros_n_, 
                                              &one, descr_l, vals_.d, mat_str_->row_ptr_.d, mat_str_->col_ind_no_borders_.d, 
                                              mat_str_->block_row_size_, info_l_, tmp1_.ptr(), y.ptr(), policy, p_buf_l_) );

    //calc solution for colors>0 - border elements

    for (int color = 1; color < mat_str_->colors_n_;++color) {
        for_each_( brs_matrix_update_colored_func<T>( color, false, mat_str_->colors_.d, 
                                                      mat_str_->row_ptr_.d, mat_str_->col_ind_.d, vals_.d,
                                                      mat_str_->block_row_size_, x.ptr(), y.ptr() ),
                  0, mat_str_->loc_rows_n_ );

        if (color+1 < mat_str_->colors_n_) {
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
            mat_str_->colors_distributors_[color-1].set_data(y.ptr(), y.size(), y.block_size(), false);
            mat_str_->colors_distributors_[color-1].sync();
#endif
        }
    }
}

template<class T>
struct brs_matrix_copy_block_diag_func
{
    FOR_EACH_FUNC_PARAMS_HELP(brs_matrix_copy_block_diag_func,
                              const int*, row_ptrs, const int*, col_inds, const T*, vals,
                              int, block_size, T*, res_vals)

    __DEVICE_TAG__ void operator()(const int &row)const
    {
        int     ptr0 = row_ptrs[row],
                ptr1 = row_ptrs[row+1];
        int     diag_ptr;  //ISSUE set some uninited value??

        for (int ptr = ptr0;ptr < ptr1;++ptr) {
            int col = col_inds[ptr];
            if (row == col) diag_ptr = ptr;
        }

        #pragma unroll
        for (int ii1 = 0;ii1 < BRS_MATRIX_MAX_BLOCKDIM;++ii1) {
            if (!(ii1 < block_size)) break;
            #pragma unroll
            for (int ii2 = 0;ii2 < BRS_MATRIX_MAX_BLOCKDIM;++ii2) {
                if (!(ii2 < block_size)) break;
                res_vals[row*block_size*block_size+BRS_MATRIX_BLK_IND(ii1,ii2)] = vals[diag_ptr*block_size*block_size+BRS_MATRIX_BLK_IND(ii1,ii2)];
            }
        }
    }

};

template<class T>
struct brs_matrix_copy_inverted_block_diag_func
{
    FOR_EACH_FUNC_PARAMS_HELP(brs_matrix_copy_inverted_block_diag_func,
                              const int*, row_ptrs, const int*, col_inds, const T*, vals,
                              int, block_size, T*, res_vals)

    __DEVICE_TAG__ void operator()(const int &row)const
    {
        int     ptr0 = row_ptrs[row],
                ptr1 = row_ptrs[row+1];
        int     diag_ptr;  //ISSUE set some uninited value??

        for (int ptr = ptr0;ptr < ptr1;++ptr) {
            int col = col_inds[ptr];
            if (row == col) diag_ptr = ptr;
        }

        T m[BRS_MATRIX_MAX_BLOCKDIM][BRS_MATRIX_MAX_BLOCKDIM*2];

        #pragma unroll
        for (int ii1 = 0;ii1 < BRS_MATRIX_MAX_BLOCKDIM;++ii1) {
            if (!(ii1 < block_size)) break;
            #pragma unroll
            for (int ii2 = 0;ii2 < BRS_MATRIX_MAX_BLOCKDIM;++ii2) {
                if (!(ii2 < block_size)) break;
                m[ii1][ii2] = vals[diag_ptr*block_size*block_size+BRS_MATRIX_BLK_IND(ii1,ii2)];
                m[ii1][ii2+BRS_MATRIX_MAX_BLOCKDIM] = (ii1 == ii2 ? T(1.f) : T(0.f));
            }
        }

        //forward step
        #pragma unroll
        for (int ii1 = 0;ii1 < BRS_MATRIX_MAX_BLOCKDIM;++ii1) {
            if (!(ii1 < block_size)) break;
            T diag = m[ii1][ii1];
            #pragma unroll
            for (int ii3 = 0;ii3 < BRS_MATRIX_MAX_BLOCKDIM;++ii3) {
                if (!(ii3 < block_size)) break;
                m[ii1][ii3] /= diag;
                m[ii1][ii3+BRS_MATRIX_MAX_BLOCKDIM] /= diag;
            }

            #pragma unroll
            for (int ii2 = 0;ii2 < BRS_MATRIX_MAX_BLOCKDIM;++ii2) {
                if (ii2 <= ii1) continue;
                if (!(ii2 < block_size)) break;
                T mul = m[ii2][ii1];
                #pragma unroll
                for (int ii3 = 0;ii3 < BRS_MATRIX_MAX_BLOCKDIM;++ii3) {
                    if (!(ii3 < block_size)) break;
                    m[ii2][ii3] -= mul*m[ii1][ii3];
                    m[ii2][ii3+BRS_MATRIX_MAX_BLOCKDIM] -= mul*m[ii1][ii3+BRS_MATRIX_MAX_BLOCKDIM];
                }
            }
        }

        //backward step
        #pragma unroll
        for (int ii1 = BRS_MATRIX_MAX_BLOCKDIM-1;ii1 >= 0;--ii1) {
            if (!(ii1 < block_size)) continue;
            #pragma unroll
            for (int ii2 = BRS_MATRIX_MAX_BLOCKDIM-1;ii2 >= 0;--ii2) {
                if (ii2 >= ii1) continue;
                T mul = m[ii2][ii1];
                #pragma unroll
                for (int ii3 = 0;ii3 < BRS_MATRIX_MAX_BLOCKDIM;++ii3) {
                    if (!(ii3 < block_size)) break;
                    m[ii2][ii3] -= mul*m[ii1][ii3];
                    m[ii2][ii3+BRS_MATRIX_MAX_BLOCKDIM] -= mul*m[ii1][ii3+BRS_MATRIX_MAX_BLOCKDIM];
                }
            }
        }

        //write result
        #pragma unroll
        for (int ii1 = 0;ii1 < BRS_MATRIX_MAX_BLOCKDIM;++ii1) {
            if (!(ii1 < block_size)) break;
            #pragma unroll
            for (int ii2 = 0;ii2 < BRS_MATRIX_MAX_BLOCKDIM;++ii2) {
                if (!(ii2 < block_size)) break;
                res_vals[row*block_size*block_size+BRS_MATRIX_BLK_IND(ii1,ii2)] = m[ii1][ii2+BRS_MATRIX_MAX_BLOCKDIM];
            }
        }
    }

};

template<class T,t_tensor_field_storage storage,class Map>
void brs_matrix<T,storage,Map>::copy_diagonal(brs_matrix &diag_mat)const
{
    //TODO checks for matrix parameters validity
    for_each_( brs_matrix_copy_block_diag_func<T>( mat_str_->row_ptr_.d, mat_str_->col_ind_.d, vals_.d,
                                                   mat_str_->block_row_size_, diag_mat.vals_.d ),
               0, mat_str_->loc_rows_n_ );
}

template<class T,t_tensor_field_storage storage,class Map>
void brs_matrix<T,storage,Map>::copy_inverted_diagonal(brs_matrix &diag_mat)const
{
    //TODO checks for matrix parameters validity
    for_each_( brs_matrix_copy_inverted_block_diag_func<T>( mat_str_->row_ptr_.d, mat_str_->col_ind_.d, vals_.d,
                                                            mat_str_->block_row_size_, diag_mat.vals_.d ),
               0, mat_str_->loc_rows_n_ );
}

template<class T>
struct brs_matrix_left_mul_by_diagonal_matrix_func
{
    FOR_EACH_FUNC_PARAMS_HELP(brs_matrix_left_mul_by_diagonal_matrix_func,
                              const T*, diag_vals, 
                              const int*, row_ptrs, const int*, col_inds, T*, vals,
                              int, block_size)

    __DEVICE_TAG__ void operator()(const int &row)const
    {
        int     ptr0 = row_ptrs[row],
                ptr1 = row_ptrs[row+1];

        T       inv_diag_m[BRS_MATRIX_MAX_BLOCKDIM][BRS_MATRIX_MAX_BLOCKDIM],
                m[BRS_MATRIX_MAX_BLOCKDIM][BRS_MATRIX_MAX_BLOCKDIM];

        #pragma unroll
        for (int ii1 = 0;ii1 < BRS_MATRIX_MAX_BLOCKDIM;++ii1) {
            if (!(ii1 < block_size)) break;
            #pragma unroll
            for (int ii2 = 0;ii2 < BRS_MATRIX_MAX_BLOCKDIM;++ii2) {
                if (!(ii2 < block_size)) break;
                inv_diag_m[ii1][ii2] = diag_vals[row*block_size*block_size+BRS_MATRIX_BLK_IND(ii1,ii2)];
            }
        }

        for (int ptr = ptr0;ptr < ptr1;++ptr) {
            #pragma unroll
            for (int ii1 = 0;ii1 < BRS_MATRIX_MAX_BLOCKDIM;++ii1) {
                if (!(ii1 < block_size)) break;
                #pragma unroll
                for (int ii2 = 0;ii2 < BRS_MATRIX_MAX_BLOCKDIM;++ii2) {
                    if (!(ii2 < block_size)) break;
                    m[ii1][ii2] = vals[ptr*block_size*block_size+BRS_MATRIX_BLK_IND(ii1,ii2)];
                }
            }
            #pragma unroll
            for (int ii1 = 0;ii1 < BRS_MATRIX_MAX_BLOCKDIM;++ii1) {
                if (!(ii1 < block_size)) break;
                #pragma unroll
                for (int ii2 = 0;ii2 < BRS_MATRIX_MAX_BLOCKDIM;++ii2) {
                    if (!(ii2 < block_size)) break;
                    T res(0.f);
                    #pragma unroll
                    for (int ii3 = 0;ii3 < BRS_MATRIX_MAX_BLOCKDIM;++ii3) {
                        if (!(ii3 < block_size)) break;
                        res += inv_diag_m[ii1][ii3]*m[ii3][ii2];
                    }
                    vals[ptr*block_size*block_size+BRS_MATRIX_BLK_IND(ii1,ii2)] = res;
                }
            }
        }
    }

};

template<class T,t_tensor_field_storage storage,class Map>
void brs_matrix<T,storage,Map>::left_mul_by_diagonal_matrix(const brs_matrix &diag_mat)
{
    //TODO checks for matrix parameters validity
    for_each_( brs_matrix_left_mul_by_diagonal_matrix_func<T>( diag_mat.vals_.d, 
                                                               mat_str_->row_ptr_.d, mat_str_->col_ind_.d, vals_.d,
                                                               mat_str_->block_row_size_ ),
               0, mat_str_->loc_rows_n_ );
}

template<class T>
struct brs_matrix_add_mul_ident_matrix_func
{
    FOR_EACH_FUNC_PARAMS_HELP(brs_matrix_add_mul_ident_matrix_func,
                              T, alpha, T, beta,  
                              const int*, row_ptrs, const int*, col_inds, T*, vals,
                              int, block_size)

    __DEVICE_TAG__ void operator()(const int &row)const
    {
        int     ptr0 = row_ptrs[row],
                ptr1 = row_ptrs[row+1];

        for (int ptr = ptr0;ptr < ptr1;++ptr) {
            int col = col_inds[ptr];

            #pragma unroll
            for (int ii1 = 0;ii1 < BRS_MATRIX_MAX_BLOCKDIM;++ii1) {
                if (!(ii1 < block_size)) break;
                #pragma unroll
                for (int ii2 = 0;ii2 < BRS_MATRIX_MAX_BLOCKDIM;++ii2) {
                    if (!(ii2 < block_size)) break;
                    vals[ptr*block_size*block_size+BRS_MATRIX_BLK_IND(ii1,ii2)] = 
                        alpha*(ii1 == ii2 ? T(1.f) : T(0.f))*(row == col ? T(1.f) : T(0.f)) + 
                        beta*vals[ptr*block_size*block_size+BRS_MATRIX_BLK_IND(ii1,ii2)];
                }
            }
        }
    }

};

template<class T,t_tensor_field_storage storage,class Map>
void brs_matrix<T,storage,Map>::add_mul_ident_matrix(T alpha, T beta)
{
    for_each_( brs_matrix_add_mul_ident_matrix_func<T>( alpha, beta, 
                                                        mat_str_->row_ptr_.d, mat_str_->col_ind_.d, vals_.d,
                                                        mat_str_->block_row_size_ ),
               0, mat_str_->loc_rows_n_ );
}

}

#endif
