#ifndef __FOR_EACH_NDIM_H__
#define __FOR_EACH_NDIM_H__

#include "for_each_config.h"
#include <vecs_mats/t_rect_tml.h>
#include "for_each_enums.h"

//ISSUE think about different interface to FUNC_T. Instead of passing idx variable may be it is better to pass some kind of iterface (which calc it on fly - to prevent extra registers pressure)

//T is ordinal type (like int)
//SERIAL_CPU realization is default
//WTF!!! why default parameter is not working
//template<t_for_each_ndim_type type = FET_SERIAL_CPU, int dim = 1, class T = int>
template<t_for_each_ndim_type type, int dim, class T = int>
struct for_each_ndim
{
	inline bool next(const t_rect_tml<T, dim> &range, t_vec_tml<T, dim> &idx)const
	{
		for (int j = dim-1;j >= 0;--j) {
			++(idx[j]);
			if (idx[j] < range.i2[j]) return true;
			idx[j] = range.i1[j];
		}
		return false;
	}

	//FUNC_T concept:
	//TODO
        //copy-constructable
	template<class FUNC_T>
	void operator()(FUNC_T f, const t_rect_tml<T, dim> &range)const
	{
		t_vec_tml<T, dim> idx = range.i1;
		do {
			f(idx);
		} while (next(range, idx));
	}
};

template<int dim, class T>
struct for_each_ndim<FET_CUDA,dim,T>
{
	//t_rect_tml<T, dim> block_size;	//ISSUE remove from here somehow: we need it just 4 cuda
	int block_size;

	template<class FUNC_T>
	void operator()(const FUNC_T &f, const t_rect_tml<T, dim> &range)const;
};

template<int dim, class T>
struct for_each_ndim<FET_OPENMP,dim,T>
{
	for_each_ndim() : threads_num(-1) {}
	//t_rect_tml<T, dim> block_size;	//ISSUE remove from here somehow: we need it just 4 cuda
	int threads_num;

	template<class FUNC_T>
	void operator()(FUNC_T f, const t_rect_tml<T, dim> &range)const;
};

#endif
