#ifndef __FOR_EACH_1D_OPENMP_IMPL_H__
#define __FOR_EACH_1D_OPENMP_IMPL_H__

//for_each_1d implementation for OPENMP case

#include "for_each_config.h"
#include <omp.h>
#include "for_each_1d.h"

template<class T>
template<class FUNC_T>
void for_each_1d<FET_OPENMP,T>::operator()(FUNC_T f, int i1, int i2)const
{	
	int real_threads_num = threads_num;
	if (threads_num < 0) real_threads_num = omp_get_max_threads();

        #pragma omp parallel for num_threads(real_threads_num)
	for (int i = i1;i < i2;++i) {
		f(i);
	}
}

template<class T>
void for_each_1d<FET_OPENMP,T>::wait()const
{
}

#endif
