#ifndef __FOR_EACH_1D_CUDA_IMPL_CUH__
#define __FOR_EACH_1D_CUDA_IMPL_CUH__

//for_each_1d implementation for CUDA case

#include <cuda_runtime.h>
#include "for_each_config.h"
#include "for_each_1d.h"

template<class FUNC_T, class T>
__global__ void ker_for_each_1d(FUNC_T f, int i1, int i2)
{
	int i = i1 + blockIdx.x*blockDim.x + threadIdx.x;
	if (!((i >= i1)&&(i < i2))) return;
	f(i);
}

template<class T>
template<class FUNC_T>
void for_each_1d<FET_CUDA,T>::operator()(FUNC_T f, int i1, int i2)const
{
	int total_sz = i2-i1;
	ker_for_each_1d<FUNC_T,T><<<(total_sz/block_size)+1,block_size>>>(f, i1, i2);
}

template<class T>
void for_each_1d<FET_CUDA,T>::wait()const
{
        //TODO error check?
        cudaStreamSynchronize(0);
}

#endif
