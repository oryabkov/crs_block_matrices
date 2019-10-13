#ifndef __RECT3_H__
#define __RECT3_H__

#include <vecs_mats/t_vec_tml.h>

#ifndef __CUDACC__
#define __DEVICE_TAG__
#else
#define __DEVICE_TAG__ __device__ __host__
#endif

//NOTE T is supposed to be int or other ordinal (not real number)
template<class T,int dim>
struct t_rect_tml
{
	t_vec_tml<T,dim>  i1, i2;
	__DEVICE_TAG__ t_rect_tml() { }
    __DEVICE_TAG__ t_rect_tml(const t_vec_tml<T,dim> &_i1, const t_vec_tml<T,dim> &_i2) : i1(_i1), i2(_i2) { }

    __DEVICE_TAG__ bool is_own(const t_vec_tml<T,dim> &p)const
    {
		for (int j = 0;j < dim;++j)
			if (!((i1[j]<=p[j])&&(p[j]<i2[j]))) return false;
		return true;
	}
    __DEVICE_TAG__ t_vec_tml<T,dim>  calc_size()const { return i2-i1; }
    __DEVICE_TAG__ T 				 calc_area()const 
    { 
    	T 	res(1);
		for (int j = 0;j < dim;++j)
			res *= (i2[j] - i1[j]);
		return res;
    }
	__DEVICE_TAG__ t_rect_tml<T,dim> intersect(const t_rect_tml<T,dim> &r)
    {
		//TODO
	}

	__DEVICE_TAG__ bool			is_empty()const
	{
		//TODO
		return true;
	}

	__DEVICE_TAG__ bool			bypass_start(t_vec_tml<T, dim> &idx)const
	{
		if (is_empty()) return false;
		idx = i1;
		return true;
	}
	__DEVICE_TAG__ bool			bypass_step(t_vec_tml<T, dim> &idx)const
	{
		for (int j = dim-1;j >= 0;--j) {
			++(idx[j]);
			if (idx[j] < i2[j]) return true;
			idx[j] = i1[j];
		}
		return false;
	}

	//ISSUE what about names (_-prefix)??
	//this pair is more for for-style bypass (t_idx i = r.bypass_start();r.is_own(i);r.bypass_step(i))
	__DEVICE_TAG__ t_vec_tml<T, dim>	_bypass_start()const
	{
		return i1;
	}
	__DEVICE_TAG__ void					_bypass_step(t_vec_tml<T, dim> &idx)const
	{
		for (int j = dim-1;j >= 0;--j) {
			++(idx[j]);
			if (idx[j] < i2[j]) return;
			if (j == 0) return;  //we are over rect, so we leave idx[0] to be out of range which can be checked by is_own(idx)
			idx[j] = i1[j];
		}
		//we are never not here
	}


};

#endif
