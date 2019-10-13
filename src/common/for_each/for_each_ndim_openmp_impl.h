#ifndef __FOR_EACH_NDIM_OPENMP_IMPL_H__
#define __FOR_EACH_NDIM_OPENMP_IMPL_H__

//for_each_ndim implementation for OPENMP case

#include "for_each_config.h"
#include <omp.h>
#include <vecs_mats/t_vec_tml.h>
#include "for_each_ndim.h"

template<int dim, class T>
template<class FUNC_T>
void for_each_ndim<FET_OPENMP,dim,T>::operator()(FUNC_T f, const t_rect_tml<T, dim> &range)const
{
	int total_sz = 1;
	for (int j = 0;j < dim;++j) total_sz *= (range.i2[j]-range.i1[j]);
	
	int real_threads_num = threads_num;
	if (threads_num < 0) real_threads_num = omp_get_max_threads();

        #pragma omp parallel for num_threads(real_threads_num)
	for (int i = 0;i < total_sz;++i) {
		//printf("%d %d \n", omp_get_thread_num(), omp_get_thread_num());
		t_vec_tml<T, dim> idx;
		int		  i_tmp = i;
		for (int j = 0;j < dim;++j) {
			idx[j] = range.i1[j] + i_tmp%(range.i2[j]-range.i1[j]);
			i_tmp /= (range.i2[j]-range.i1[j]);
		}
		f(idx);
	}
}

#endif
