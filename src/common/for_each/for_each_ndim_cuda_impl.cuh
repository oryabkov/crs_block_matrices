#ifndef __FOR_EACH_NDIM_CUDA_IMPL_CUH__
#define __FOR_EACH_NDIM_CUDA_IMPL_CUH__

//for_each_ndim implementation for CUDA case

#include "for_each_config.h"
#include <vecs_mats/t_vec_tml.h>
#include "for_each_ndim.h"

template<class FUNC_T, int dim, class T>
__global__ void ker_for_each(FUNC_T f, t_rect_tml<T, dim> range, int total_sz)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (!((i >= 0)&&(i < total_sz))) return;
	t_vec_tml<T, dim> idx;
	for (int j = 0;j < dim;++j) {
		idx[j] = range.i1[j] + i%(range.i2[j]-range.i1[j]);
		i /= (range.i2[j]-range.i1[j]);
	}
	f(idx);
}

/*template<int dim, class T>
struct for_each_ndim<FET_CUDA,dim,T>
{
	//t_rect_tml<T, dim> block_size;	//ISSUE remove from here somehow: we need it just 4 cuda
	int block_size;

	template<class FUNC_T>
	void operator()(const FUNC_T &f, const t_rect_tml<T, dim> &range)const
	{
		int total_sz = 1;
		for (int j = 0;j < dim;++j) total_sz *= (range.i2[j]-range.i1[j]);

                ker_for_each<FUNC_T,dim,T><<<total_sz/block_size,block_size>>>(f, range, total_sz);
	}
};*/

template<int dim, class T>
template<class FUNC_T>
void for_each_ndim<FET_CUDA,dim,T>::operator()(const FUNC_T &f, const t_rect_tml<T, dim> &range)const
{
	int total_sz = 1;
	for (int j = 0;j < dim;++j) total_sz *= (range.i2[j]-range.i1[j]);
	
	ker_for_each<FUNC_T,dim,T><<<(total_sz/block_size)+1,block_size>>>(f, range, total_sz);
}

#endif
