#ifndef __T_TENSOR_FIELD_TML_H__
#define __T_TENSOR_FIELD_TML_H__

#include "tensor_field_config.h"
#include <cassert>
#include <vecs_mats/t_vec_tml.h>
#include <utils/device_tag.h>
#include "tensor_field_enums.h"
#include "tensor_field_mem_funcs.h"

//GPU-oriented classes

/*#ifndef __CUDACC__
#define __DEVICE_TAG__
#else
#define __DEVICE_TAG__ __device__ __host__
#endif

#ifndef __CUDACC__
#define __DEVICE_ONLY_TAG__
#else
#define __DEVICE_ONLY_TAG__ __device__
#endif*/

template<typename T,t_tensor_field_storage storage = TFS_DEVICE>
struct t_tensor0_field_tml;
template<typename T,t_tensor_field_storage storage = TFS_DEVICE>
struct t_const_tensor0_field_tml;
template<typename T,t_tensor_field_storage filed_storage = TFS_DEVICE>
struct t_tensor0_field_view_tml;

template<typename T,int dim1, t_tensor_field_storage storage = TFS_DEVICE, t_tensor_field_arrangement arrangement = t_tensor_field_arrangement_type_helper<storage>::arrangement >
struct t_tensor1_field_tml;
template<typename T,int dim1, t_tensor_field_storage storage = TFS_DEVICE, t_tensor_field_arrangement arrangement = t_tensor_field_arrangement_type_helper<storage>::arrangement >
struct t_const_tensor1_field_tml;
template<typename T,int dim1, t_tensor_field_storage filed_storage = TFS_DEVICE, t_tensor_field_arrangement filed_arrangement = t_tensor_field_arrangement_type_helper<filed_storage>::arrangement >
struct t_tensor1_field_view_tml;

template<typename T,int dim1,int dim2, t_tensor_field_storage storage = TFS_DEVICE, t_tensor_field_arrangement arrangement = t_tensor_field_arrangement_type_helper<storage>::arrangement >
struct t_tensor2_field_tml;
template<typename T,int dim1,int dim2, t_tensor_field_storage storage = TFS_DEVICE, t_tensor_field_arrangement arrangement = t_tensor_field_arrangement_type_helper<storage>::arrangement >
struct t_const_tensor2_field_tml;
template<typename T,int dim1,int dim2, t_tensor_field_storage filed_storage = TFS_DEVICE, t_tensor_field_arrangement filed_arrangement = t_tensor_field_arrangement_type_helper<filed_storage>::arrangement >
struct t_tensor2_field_view_tml;

template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage = TFS_DEVICE, t_tensor_field_arrangement arrangement = t_tensor_field_arrangement_type_helper<storage>::arrangement >
struct t_tensor3_field_tml;
template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage = TFS_DEVICE, t_tensor_field_arrangement arrangement = t_tensor_field_arrangement_type_helper<storage>::arrangement >
struct t_const_tensor3_field_tml;
template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage filed_storage = TFS_DEVICE, t_tensor_field_arrangement filed_arrangement = t_tensor_field_arrangement_type_helper<filed_storage>::arrangement >
struct t_tensor3_field_view_tml;

template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage = TFS_DEVICE, t_tensor_field_arrangement arrangement = t_tensor_field_arrangement_type_helper<storage>::arrangement >
struct t_tensor4_field_tml;
template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage = TFS_DEVICE, t_tensor_field_arrangement arrangement = t_tensor_field_arrangement_type_helper<storage>::arrangement >
struct t_const_tensor4_field_tml;
template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage filed_storage = TFS_DEVICE, t_tensor_field_arrangement filed_arrangement = t_tensor_field_arrangement_type_helper<filed_storage>::arrangement >
struct t_tensor4_field_view_tml;

//  ------------------------------------------------ TENSOR0 -----------------------------------------------------------

template<typename T,t_tensor_field_storage storage>
struct t_tensor0_field_base_tml
{
	//ISSUE maybe add something like following??
	//static const int				st_sz1 = 1;
	typedef T					value_type;
	typedef t_tensor0_field_view_tml<T,storage>	view_type;

	int	N;		//'big',dynamic dimension
	T	*d;
	bool	own;    	//whether we should free d-pointed memory when this object dies, for passing in cuda kernels
	int	logic_i0;       //starting logical index for multigpu support (default - 0) (i.e. we enumerate elements of field not from 0 but from logic_i0)

	__DEVICE_TAG__ int size()const { return N; }
	__DEVICE_TAG__ int total_size()const { return size(); }

	__DEVICE_TAG__ t_tensor0_field_base_tml() : d(NULL) {}
	__DEVICE_TAG__ t_tensor0_field_base_tml(const t_tensor0_field_base_tml &tf) { *this = tf; own = false; }
	__DEVICE_TAG__ ~t_tensor0_field_base_tml()
	{
		#ifndef __CUDA_ARCH__
		if ((d != NULL) && own) free();
		#endif
	}
	void	init(int _N, int _logic_i0 = 0)
	{
		logic_i0 = _logic_i0;
		N = _N;
		own = true;
		tensor_field_malloc<storage>((void**)&d, sizeof(T)*N);
		assert((d != NULL) && own);
	}
	void	free()
	{
		assert((d != NULL) && own);
		tensor_field_free<storage>(d);
		d = NULL;
		//printf("tensor0:free\n");
	}

protected:
	__DEVICE_TAG__ void			assign(const t_tensor0_field_base_tml &tf) { *this = tf; own = false; }

	__DEVICE_TAG__ T			&operator()(int i_logic)const { int i = i_logic - logic_i0; return d[i]; }
};

template<typename T,t_tensor_field_storage storage>
struct t_tensor0_field_tml : public  t_tensor0_field_base_tml<T,storage>
{
	typedef t_tensor0_field_base_tml<T,storage>		t_parent;

	__DEVICE_TAG__ t_tensor0_field_tml() : t_parent() { }
	__DEVICE_TAG__ t_tensor0_field_tml(const t_tensor0_field_tml &tf) : t_parent(tf) { }

	__DEVICE_TAG__ t_tensor0_field_tml	&operator=(const t_tensor0_field_tml &tf) { t_parent::assign(tf); return *this; }

	using 			t_parent::operator();
	
	//some adapters
	t_tensor1_field_tml<T,1,storage>	as_tensor1()const;
	t_tensor2_field_tml<T,1,1,storage>	as_tensor2()const;
	t_tensor3_field_tml<T,1,1,1,storage>	as_tensor3()const;
	t_tensor4_field_tml<T,1,1,1,1,storage>	as_tensor4()const;
};

template<typename T,t_tensor_field_storage storage>
struct t_const_tensor0_field_tml : public  t_tensor0_field_base_tml<T,storage>
{
	typedef t_tensor0_field_base_tml<T,storage>		t_parent;

	__DEVICE_TAG__ t_const_tensor0_field_tml() : t_parent() { }
	//we can create constant field from non-constant
	__DEVICE_TAG__ t_const_tensor0_field_tml(const t_tensor0_field_tml<T,storage> &tf) : t_parent(tf) { }
	__DEVICE_TAG__ t_const_tensor0_field_tml(const t_const_tensor0_field_tml &tf) : t_parent(tf) { }
	
	__DEVICE_TAG__ t_const_tensor0_field_tml	&operator=(const t_tensor0_field_tml<T,storage> &tf) { t_parent::assign(tf); return *this; }
	__DEVICE_TAG__ t_const_tensor0_field_tml	&operator=(const t_const_tensor0_field_tml &tf) { t_parent::assign(tf); return *this; }

	__DEVICE_TAG__ const  T	&operator()(int i_logic)const { return t_parent::operator()(i_logic); }
	
	//some adapters
	t_const_tensor1_field_tml<T,1,storage>		as_tensor1()const;
	t_const_tensor2_field_tml<T,1,1,storage>	as_tensor2()const;
	t_const_tensor3_field_tml<T,1,1,1,storage>	as_tensor3()const;
	t_const_tensor4_field_tml<T,1,1,1,1,storage>	as_tensor4()const;
};

//field_storage refers not to view storage but rather to viewed field storage
template<typename T, t_tensor_field_storage field_storage>
struct t_tensor0_field_view_tml : public t_tensor0_field_tml<T,TFS_HOST>
{
	typedef t_tensor0_field_tml<T,field_storage>	field_t;
	//*_type nested type more like something 'official'; t_* are for internal use, but i leave them in public (no hurt from that)
	//TODO but! why field_t?? i think t_field should be instead
	typedef field_t					field_type;
	typedef t_tensor0_field_tml<T,TFS_HOST>		t_parent;
	const field_t	*f;

	t_tensor0_field_view_tml() : f(NULL) {}
	t_tensor0_field_view_tml(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, int _logic_i0 = 0) : f(NULL)
	{
		init(_f, copy_data, indexing, _logic_i0);
        }
	void	init(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, int _logic_i0 = 0)
	{
		assert(f == NULL);
		f = &_f;
		t_parent::init(f->N, (indexing == TVI_NATIVE?f->logic_i0:_logic_i0));
		if (copy_data) {
			tensor_field_copy_storage_2_host<field_storage>(t_parent::d, f->d, sizeof(T)*t_parent::total_size());
		}
	}
	void	release(bool copy_data = true)
	{
		assert(f != NULL);
		if (copy_data) {
			tensor_field_copy_host_2_storage<field_storage>(f->d, t_parent::d, sizeof(T)*t_parent::total_size());
		}
		this->free();
		f = NULL;
	}

	//ISSUE not sure about this
	int	begin()const { return t_parent::logic_i0; }
	int	end()const { return t_parent::logic_i0+t_parent::size(); }
	
	//TODO operator= should be hidden
private:
	t_tensor0_field_view_tml	&operator=(const t_tensor0_field_view_tml &tf_view) { return *this; }	//just to avoid warning
};

//  ------------------------------------------------ TENSOR1 -----------------------------------------------------------

template<int dim1, t_tensor_field_arrangement arrangement>
struct t_tensor1_field_index_base_tml
{
        __DEVICE_TAG__ int calc_idx(int N, int i, int i1)const { return i1*N + i; }
};

template<int dim1>
struct t_tensor1_field_index_base_tml<dim1, TFA_CPU_STYLE>
{
        __DEVICE_TAG__ int calc_idx(int N, int i, int i1)const { return i*dim1 + i1; }
};

//dim1 - 'small', static dimension
template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
struct t_tensor1_field_base_tml : public t_tensor1_field_index_base_tml<dim1,arrangement>
{
        typedef t_tensor1_field_index_base_tml<dim1,arrangement>        t_parent;

	//st here stands for static
	static const int					        st_sz1 = dim1;
	//ISSUE maybe add following indexes (st_sz2, etc) equals 1
	typedef T						        value_type;
	typedef t_tensor1_field_view_tml<T,dim1,storage,arrangement>	view_type;

	int	N;		//'big',dynamic dimension
	T	*d;
	bool	own;    	//whether we should free d-pointed memory when this object dies, for passing in cuda kernels
	int	logic_i0;       //starting logical index for multigpu support (default - 0) (i.e. we enumerate elements of field not from 0 but from logic_i0)

	__DEVICE_TAG__ int size()const { return N; }
	__DEVICE_TAG__ int total_size()const { return size()*dim1; }

	__DEVICE_TAG__ t_tensor1_field_base_tml() : d(NULL) {}
	__DEVICE_TAG__ t_tensor1_field_base_tml(const t_tensor1_field_base_tml &tf) { *this = tf; own = false; }
	__DEVICE_TAG__ ~t_tensor1_field_base_tml()
	{
		#ifndef __CUDA_ARCH__
		if ((d != NULL) && own) free();
		#endif
	}
	void	init(int _N, int _logic_i0 = 0) 
	{
		logic_i0 = _logic_i0; 
		N = _N; 
		own = true;
		tensor_field_malloc<storage>((void**)&d, sizeof(T)*N*dim1);
		assert((d != NULL) && own);
	}
	void	free()
	{
		assert((d != NULL) && own);
		tensor_field_free<storage>(d);
		d = NULL;
		//printf("tensor1:free\n");
	}

	//TODO temp
	//__DEVICE_TAG__ int size1()const { return dim1; }

protected:
	__DEVICE_TAG__ void			assign(const t_tensor1_field_base_tml &tf) { *this = tf; own = false; }

	__DEVICE_TAG__ T			&operator()(int i_logic,int i1)const { int i = i_logic - logic_i0; return d[t_parent::calc_idx(N, i, i1)]; }
	//VEC concept
	//TODO
	template<class VEC>
	__DEVICE_TAG__ VEC			get(int i)const
	{
		VEC	res;
		for (int j = 0;j < dim1;++j) res[j] = (*this)(i,j);
		return res;
	}
	template<class VEC>
	__DEVICE_TAG__ void			get(int i, VEC &res)const
	{
		for (int j = 0;j < dim1;++j) res[j] = (*this)(i,j);
	}
	template<class VEC>
	__DEVICE_TAG__ void			set(int i, const VEC &v)const
	{
		for (int j = 0;j < dim1;++j) (*this)(i,j) = v[j];
	}
	//TODO not best way written (index calc duplication, etc)
	template<class VEC>
	void					host_get(int i_logic, VEC &res)const
	{
		//ISSUE hmm is it safe??
		for (int j = 0;j < dim1;++j) {
			tensor_field_copy_storage_2_host<storage>(&(res[j]), &((*this)(i_logic,j)), sizeof(T));
		}
	}
	__DEVICE_TAG__ t_vec_tml<T,dim1>	getv(int i)const { return get<t_vec_tml<T,dim1> >(i); }
	__DEVICE_TAG__ void			getv(int i, t_vec_tml<T,dim1> &res)const { get(i, res); }
	__DEVICE_TAG__ void			setv(int i, const t_vec_tml<T,dim1> &v)const { set(i, v); }
	void					host_getv(int i_logic, t_vec_tml<T,dim1> &res)const { host_get(i_logic, res); }

	//TODO temporal solution!
	__DEVICE_TAG__ void			get_mat33(int i, T res[3][3])const
	{
		for (int j = 0;j < 3;++j)
		for (int k = 0;k < 3;++k)
			res[j][k] = (*this)(i,j*3+k);
	}
	//TODO temporal solution!
	__DEVICE_TAG__ void			set_mat33(int i, T m[3][3])const
	{
		for (int j = 0;j < 3;++j)
		for (int k = 0;k < 3;++k)
			(*this)(i, j*3+k) = m[j][k];
	}
};

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
struct t_tensor1_field_tml : public  t_tensor1_field_base_tml<T,dim1,storage,arrangement>
{
	typedef t_tensor1_field_base_tml<T,dim1,storage,arrangement>		t_parent;

	__DEVICE_TAG__ t_tensor1_field_tml() : t_parent() { }
	__DEVICE_TAG__ t_tensor1_field_tml(const t_tensor1_field_tml &tf) : t_parent(tf) { }
	
	__DEVICE_TAG__ t_tensor1_field_tml	&operator=(const t_tensor1_field_tml &tf) { t_parent::assign(tf); return *this; }

	using 			t_parent::operator();
	using 			t_parent::get;
	using 			t_parent::set;
	using 			t_parent::host_get;
	using 			t_parent::getv;
	using 			t_parent::setv;
	using 			t_parent::host_getv;
	using 			t_parent::get_mat33;
	using 			t_parent::set_mat33;
	
	//some adapters
	t_tensor2_field_tml<T,1,dim1,storage,arrangement>		as_tensor2_f()const;
	t_tensor2_field_tml<T,dim1,1,storage,arrangement>		as_tensor2_b()const;
	t_tensor3_field_tml<T,1,1,dim1,storage,arrangement>		as_tensor3_f()const;
	t_tensor3_field_tml<T,dim1,1,1,storage,arrangement>		as_tensor3_b()const;
	t_tensor4_field_tml<T,1,1,1,dim1,storage,arrangement>	        as_tensor4_f()const;
	t_tensor4_field_tml<T,dim1,1,1,1,storage,arrangement>	        as_tensor4_b()const;
};

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
struct t_const_tensor1_field_tml : public  t_tensor1_field_base_tml<T,dim1,storage,arrangement>
{
	typedef t_tensor1_field_base_tml<T,dim1,storage,arrangement>		t_parent;

	__DEVICE_TAG__ t_const_tensor1_field_tml() : t_parent() { }
	//we can create constant field from non-constant
	__DEVICE_TAG__ t_const_tensor1_field_tml(const t_tensor1_field_tml<T,dim1,storage> &tf) : t_parent(tf) { }
	__DEVICE_TAG__ t_const_tensor1_field_tml(const t_const_tensor1_field_tml &tf) : t_parent(tf) { }

	__DEVICE_TAG__ t_const_tensor1_field_tml	&operator=(const t_tensor1_field_tml<T,dim1,storage> &tf) { t_parent::assign(tf); return *this; }
	__DEVICE_TAG__ t_const_tensor1_field_tml	&operator=(const t_const_tensor1_field_tml &tf) { t_parent::assign(tf); return *this; }

	__DEVICE_TAG__ const  T	&operator()(int i_logic,int i1)const { return t_parent::operator()(i_logic,i1); }
	using 			t_parent::get;
	using 			t_parent::host_get;
	using 			t_parent::getv;
	using 			t_parent::host_getv;
	using 			t_parent::get_mat33;
	
	//some adapters
	t_const_tensor2_field_tml<T,1,dim1,storage,arrangement>	        as_tensor2_f()const;
	t_const_tensor2_field_tml<T,dim1,1,storage,arrangement>	        as_tensor2_b()const;
	t_const_tensor3_field_tml<T,1,1,dim1,storage,arrangement>	as_tensor3_f()const;
	t_const_tensor3_field_tml<T,dim1,1,1,storage,arrangement>	as_tensor3_b()const;
	t_const_tensor4_field_tml<T,1,1,1,dim1,storage,arrangement>	as_tensor4_f()const;
	t_const_tensor4_field_tml<T,dim1,1,1,1,storage,arrangement>	as_tensor4_b()const;
};

//field_storage refers not to view storage but rather to viewed field storage
template<typename T, int dim1, t_tensor_field_storage field_storage, t_tensor_field_arrangement field_arrangement>
struct t_tensor1_field_view_tml : public t_tensor1_field_tml<T,dim1,TFS_HOST,field_arrangement>
{
	typedef t_tensor1_field_tml<T,dim1,field_storage,field_arrangement>	field_t;
	//see t_tensor0_field_view_tml comments
	typedef field_t						                field_type;
	typedef t_tensor1_field_tml<T,dim1,TFS_HOST,field_arrangement>		t_parent;
	const field_t	*f;

	t_tensor1_field_view_tml() : f(NULL) {}
	t_tensor1_field_view_tml(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, int _logic_i0 = 0) : f(NULL)
	{
		init(_f, copy_data, indexing, _logic_i0);
        }
	void	init(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, int _logic_i0 = 0)
	{
		assert(f == NULL);
		f = &_f;
		t_parent::init(f->N, (indexing == TVI_NATIVE?f->logic_i0:_logic_i0));
		if (copy_data) {
			tensor_field_copy_storage_2_host<field_storage>(t_parent::d, f->d, sizeof(T)*t_parent::total_size());
		}
	}
	void	release(bool copy_data = true)
	{
		assert(f != NULL);
		if (copy_data) {
			tensor_field_copy_host_2_storage<field_storage>(f->d, t_parent::d, sizeof(T)*t_parent::total_size());
		}
		this->free();
		f = NULL;
	}
	
	//ISSUE not sure about this
	int	begin()const { return t_parent::logic_i0; }
	int	end()const { return t_parent::logic_i0+t_parent::size(); }
	
private:
	t_tensor1_field_view_tml	&operator=(const t_tensor1_field_view_tml &tf_view) { return *this; }	//just to avoid warning
};

//  ------------------------------------------------ TENSOR2 -----------------------------------------------------------

template<int dim1,int dim2, t_tensor_field_arrangement arrangement>
struct t_tensor2_field_index_base_tml
{
        __DEVICE_TAG__ int calc_idx(int N, int i, int i1, int i2)const { return i1*dim2*N + i2*N + i; }
};

template<int dim1,int dim2>
struct t_tensor2_field_index_base_tml<dim1, dim2, TFA_CPU_STYLE>
{
        __DEVICE_TAG__ int calc_idx(int N, int i, int i1, int i2)const { return i*dim1*dim2 + i1*dim2 + i2; }
};

//dim1,dim2 - 'small', static dimensions
template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
struct t_tensor2_field_base_tml : public t_tensor2_field_index_base_tml<dim1,dim2,arrangement>
{
        typedef t_tensor2_field_index_base_tml<dim1,dim2,arrangement>       t_parent;

	//st here stands for static
	static const int					            st_sz1 = dim1;
	static const int					            st_sz2 = dim2;
	//ISSUE maybe add following indexes (st_sz2, etc) equals 1
	typedef T						            value_type;
	typedef t_tensor2_field_view_tml<T,dim1,dim2,storage,arrangement>   view_type;

	int	N;		//'big',dynamic dimension
	T	*d;
	bool	own;    	//whether we should free d-pointed memory when this object dies, for passing in cuda kernels
	int	logic_i0;       //starting logical index for multigpu support (default - 0) (i.e. we enumerate elements of field not from 0 but from logic_i0)

        __DEVICE_TAG__ int size()const { return N; }
	__DEVICE_TAG__ int total_size()const { return size()*dim1*dim2; }

	__DEVICE_TAG__ t_tensor2_field_base_tml() : d(NULL) {}
	__DEVICE_TAG__ t_tensor2_field_base_tml(const t_tensor2_field_base_tml &tf) { *this = tf; own = false; }
	__DEVICE_TAG__ ~t_tensor2_field_base_tml()
	{
		#ifndef __CUDA_ARCH__
		if ((d != NULL) && own) free();
		#endif
	}
	void	init(int _N, int _logic_i0 = 0)
	{
		logic_i0 = _logic_i0;
		N = _N;
		own = true;
		tensor_field_malloc<storage>((void**)&d, sizeof(T)*N*dim1*dim2);
		assert((d != NULL) && own);
	}
	void	free()
	{
		assert((d != NULL) && own);
		tensor_field_free<storage>(d);
		d = NULL;
		//printf("tensor1:free\n");
	}


protected:
	__DEVICE_TAG__ void			assign(const t_tensor2_field_base_tml &tf) { *this = tf; own = false; }

	__DEVICE_TAG__ T			&operator()(int i_logic,int i1,int i2)const { int i = i_logic - logic_i0; return d[t_parent::calc_idx(N, i, i1, i2)]; }
	template<class VEC>
	__DEVICE_TAG__ VEC			get(int i, int i1)const
	{
		VEC	res;
		for (int j = 0;j < dim2;++j) res[j] = (*this)(i,i1,j);
		return res;
	}
	template<class VEC>
	__DEVICE_TAG__ void			get(int i, int i1, VEC &res)const
	{
		for (int j = 0;j < dim2;++j) res[j] = (*this)(i,i1,j);
	}
	template<class VEC>
	__DEVICE_TAG__ void			set(int i, int i1, const VEC &v)const
	{
		for (int j = 0;j < dim2;++j) (*this)(i,i1,j) = v[j];
	}
	template<class VEC>
	__DEVICE_TAG__ VEC			get(int i, char x, int i2)const
	{
		VEC	res;
		for (int j = 0;j < dim1;++j) res[j] = (*this)(i,j,i2);
		return res;
	}
	template<class VEC>
	__DEVICE_TAG__ void			get(int i, char x, int i2, VEC &res)const
	{
		for (int j = 0;j < dim1;++j) res[j] = (*this)(i,j,i2);
	}
	template<class VEC>
	__DEVICE_TAG__ void			set(int i, char x, int i2, const VEC &v)const
	{
		for (int j = 0;j < dim1;++j) (*this)(i,j,i2) = v[j];
	}
	__DEVICE_TAG__ t_vec_tml<T,dim2>	getv(int i, int i1)const { return get<t_vec_tml<T,dim2> >(i, i1); }
	__DEVICE_TAG__ void			getv(int i, int i1, t_vec_tml<T,dim2> &res)const { get(i, i1, res); }
	__DEVICE_TAG__ void			setv(int i, int i1, const t_vec_tml<T,dim2> &v)const { set(i, i1, v); }
	__DEVICE_TAG__ t_vec_tml<T,dim1>	getv(int i, char x, int i2)const { return get<t_vec_tml<T,dim1> >(i, x, i2); }
	__DEVICE_TAG__ void			getv(int i, char x, int i2, t_vec_tml<T,dim1> &res)const { get(i, x, i2, res); }
	__DEVICE_TAG__ void			setv(int i, char x, int i2, const t_vec_tml<T,dim1> &v)const { set(i, x, i2, v); }
	//TODO MAT concept
	template<class MAT>
	__DEVICE_TAG__ MAT			get2(int i)const
	{
		MAT	res;
		for (int i1 = 0;i1 < dim1;++i1)
		for (int i2 = 0;i2 < dim2;++i2)
			res(i1,i2) = (*this)(i,i1,i2);
		return res;
	}
        template<class MAT>
	__DEVICE_TAG__ void			get2(int i, MAT &res)const
	{
		for (int i1 = 0;i1 < dim1;++i1)
		for (int i2 = 0;i2 < dim2;++i2)
			res(i1,i2) = (*this)(i,i1,i2);
	}
	template<class MAT>
	__DEVICE_TAG__ void			set2(int i, MAT &m)const
	{
		for (int i1 = 0;i1 < dim1;++i1)
		for (int i2 = 0;i2 < dim2;++i2)
			(*this)(i,i1,i2) = m(i1,i2);
	}

	//TODO temporal solution!
	__DEVICE_TAG__ void			get_mat33(int i,int i1, T res[3][3])const
	{
		for (int j = 0;j < 3;++j)
		for (int k = 0;k < 3;++k)
			res[j][k] = (*this)(i, i1, j*3+k);
	}
	//TODO temporal solution!
	__DEVICE_TAG__ void			set_mat33(int i,int i1, T m[3][3])const
	{
		for (int j = 0;j < 3;++j)
		for (int k = 0;k < 3;++k)
			(*this)(i, i1, j*3+k) = m[j][k];
	}
};

//ndim is 'spartial' 'dynamic' dimension
//dim1,dim2 - 'small', static dimensions
template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
struct t_tensor2_field_tml : public t_tensor2_field_base_tml<T,dim1,dim2,storage,arrangement>
{
	typedef t_tensor2_field_base_tml<T,dim1,dim2,storage,arrangement>		t_parent;

	__DEVICE_TAG__ t_tensor2_field_tml() : t_parent() { }
	__DEVICE_TAG__ t_tensor2_field_tml(const t_tensor2_field_tml &tf) : t_parent(tf) { }
	
	__DEVICE_TAG__ t_tensor2_field_tml	&operator=(const t_tensor2_field_tml &tf) { t_parent::assign(tf); return *this; }

	using 			t_parent::operator();
	using 			t_parent::get;
	using 			t_parent::set;
	//using 			t_parent::host_get;
	using 			t_parent::getv;
	using 			t_parent::setv;
	//using 			t_parent::host_getv;
	using 			t_parent::get2;
	using 			t_parent::set2;
	using 			t_parent::get_mat33;
	using 			t_parent::set_mat33;
	
	t_tensor1_field_tml<T,dim1*dim2,storage,arrangement>	        as_tensor1()const;
	t_tensor3_field_tml<T,1,dim1,dim2,storage,arrangement>	        as_tensor3_f()const;
	t_tensor3_field_tml<T,dim1,dim2,1,storage,arrangement>	        as_tensor3_b()const;
	t_tensor4_field_tml<T,1,1,dim1,dim2,storage,arrangement>	as_tensor4_f()const;
	t_tensor4_field_tml<T,dim1,dim2,1,1,storage,arrangement>	as_tensor4_b()const;
};

template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement >
struct t_const_tensor2_field_tml : public  t_tensor2_field_base_tml<T,dim1,dim2,storage,arrangement>
{
	typedef t_tensor2_field_base_tml<T,dim1,dim2,storage,arrangement>		t_parent;

	__DEVICE_TAG__ t_const_tensor2_field_tml() : t_parent() { }
	//we can create constant field from non-constant
	__DEVICE_TAG__ t_const_tensor2_field_tml(const t_tensor2_field_tml<T,dim1,dim2,storage> &tf) : t_parent(tf) { }
	__DEVICE_TAG__ t_const_tensor2_field_tml(const t_const_tensor2_field_tml &tf) : t_parent(tf) { }
	
	__DEVICE_TAG__ t_const_tensor2_field_tml	&operator=(const t_tensor2_field_tml<T,dim1,dim2,storage> &tf) { t_parent::assign(tf); return *this; }
	__DEVICE_TAG__ t_const_tensor2_field_tml	&operator=(const t_const_tensor2_field_tml &tf) { t_parent::assign(tf); return *this; }

	__DEVICE_TAG__ const  T	&operator()(int i_logic, int i1, int i2)const { return t_parent::operator()(i_logic, i1, i2); }
	using 			t_parent::get;
	//using 			t_parent::host_get;
	using 			t_parent::getv;
	//using 			t_parent::host_getv;
	using 			t_parent::get2;
	using 			t_parent::get_mat33;
	
	t_const_tensor1_field_tml<T,dim1*dim2,storage,arrangement>	as_tensor1()const;
	t_const_tensor3_field_tml<T,1,dim1,dim2,storage,arrangement>	as_tensor3_f()const;
	t_const_tensor3_field_tml<T,dim1,dim2,1,storage,arrangement>	as_tensor3_b()const;
	t_const_tensor4_field_tml<T,1,1,dim1,dim2,storage,arrangement>	as_tensor4_f()const;
	t_const_tensor4_field_tml<T,dim1,dim2,1,1,storage,arrangement>	as_tensor4_b()const;
};

//field_storage refers not to view storage but rather to viewed field storage
template<typename T,int dim1,int dim2, t_tensor_field_storage field_storage, t_tensor_field_arrangement field_arrangement>
struct t_tensor2_field_view_tml : public t_tensor2_field_tml<T,dim1,dim2,TFS_HOST,field_arrangement>
{
	typedef t_tensor2_field_tml<T,dim1,dim2,field_storage,field_arrangement>	field_t;
	//see t_tensor0_field_view_tml comments
	typedef field_t						                        field_type;
	typedef t_tensor2_field_tml<T,dim1,dim2,TFS_HOST,field_arrangement>	        t_parent;
	const field_t	*f;

	t_tensor2_field_view_tml() : f(NULL) {}
	t_tensor2_field_view_tml(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, int _logic_i0 = 0) : f(NULL)
	{
		init(_f, copy_data, indexing, _logic_i0);
        }
	void	init(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, int _logic_i0 = 0)
	{
		assert(f == NULL);
		f = &_f;
		t_parent::init(f->N, (indexing == TVI_NATIVE?f->logic_i0:_logic_i0));
		if (copy_data) {
			tensor_field_copy_storage_2_host<field_storage>(t_parent::d, f->d, sizeof(T)*t_parent::total_size());
		}
	}
	void	release(bool copy_data = true)
	{
		assert(f != NULL);
		if (copy_data) {
			tensor_field_copy_host_2_storage<field_storage>(f->d, t_parent::d, sizeof(T)*t_parent::total_size());
		}
		this->free();
		f = NULL;
	}

	//ISSUE not sure about this
	int	begin()const { return t_parent::logic_i0; }
	int	end()const { return t_parent::logic_i0+t_parent::size(); }
	
private:
	t_tensor2_field_view_tml	&operator=(const t_tensor2_field_view_tml &tf_view) { return *this; }	//just to avoid warning
};

//  ------------------------------------------------ TENSOR3 -----------------------------------------------------------

template<int dim1,int dim2,int dim3, t_tensor_field_arrangement arrangement>
struct t_tensor3_field_index_base_tml
{
        __DEVICE_TAG__ int calc_idx(int N, int i, int i1, int i2, int i3)const { return i1*dim2*dim3*N + i2*dim3*N + i3*N + i; }
};

template<int dim1,int dim2,int dim3>
struct t_tensor3_field_index_base_tml<dim1, dim2, dim3, TFA_CPU_STYLE>
{
        __DEVICE_TAG__ int calc_idx(int N, int i, int i1, int i2, int i3)const { return i*dim1*dim2*dim3 + i1*dim2*dim3 + i2*dim3 + i3; }
};

//dim1,dim2 - 'small', static dimensions
template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
struct t_tensor3_field_base_tml : public t_tensor3_field_index_base_tml<dim1,dim2,dim3,arrangement>
{
        typedef t_tensor3_field_index_base_tml<dim1,dim2,dim3,arrangement>      t_parent;

	//st here stands for static
	static const int					                st_sz1 = dim1;
	static const int					                st_sz2 = dim2;
	static const int					                st_sz3 = dim3;
	//ISSUE maybe add following indexes (st_sz2, etc) equals 1
	typedef T							        value_type;
	typedef t_tensor3_field_view_tml<T,dim1,dim2,dim3,storage,arrangement>	view_type;

	int	N;		//'big',dynamic dimension
	T	*d;
	bool	own;    	//whether we should free d-pointed memory when this object dies, for passing in cuda kernels
	int	logic_i0;       //starting logical index for multigpu support (default - 0) (i.e. we enumerate elements of field not from 0 but from logic_i0)

        __DEVICE_TAG__ int size()const { return N; }
	__DEVICE_TAG__ int total_size()const { return size()*dim1*dim2*dim3; }

	__DEVICE_TAG__ t_tensor3_field_base_tml() : d(NULL) {}
	__DEVICE_TAG__ t_tensor3_field_base_tml(const t_tensor3_field_base_tml &tf) { *this = tf; own = false; }
	__DEVICE_TAG__ ~t_tensor3_field_base_tml()
	{
		#ifndef __CUDA_ARCH__
		if ((d != NULL) && own) free();
		#endif
	}
	void	init(int _N, int _logic_i0 = 0)
	{
		logic_i0 = _logic_i0;
		N = _N;
		own = true;
		tensor_field_malloc<storage>((void**)&d, sizeof(T)*N*dim1*dim2*dim3);
		assert((d != NULL) && own);
	}
	void	free()
	{
		assert((d != NULL) && own);
		tensor_field_free<storage>(d);
		d = NULL;
		//printf("tensor1:free\n");
	}


protected:
	__DEVICE_TAG__ void			assign(const t_tensor3_field_base_tml &tf) { *this = tf; own = false; }

	__DEVICE_TAG__ T			&operator()(int i_logic,int i1,int i2,int i3)const { int i = i_logic - logic_i0; return d[t_parent::calc_idx(N,i,i1,i2,i3)]; }
	template<class VEC>
	__DEVICE_TAG__ VEC			get(int i, int i1, int i2)const
	{
		VEC	res;
		for (int j = 0;j < dim3;++j) res[j] = (*this)(i,i1,i2,j);
		return res;
	}
	template<class VEC>
	__DEVICE_TAG__ void			get(int i, int i1, int i2, VEC &res)const
	{
		for (int j = 0;j < dim3;++j) res[j] = (*this)(i,i1,i2,j);
	}
	template<class VEC>
	__DEVICE_TAG__ void			set(int i, int i1, int i2, const VEC &v)const
	{
		for (int j = 0;j < dim3;++j) (*this)(i,i1,i2,j) = v[j];
	}
	template<class VEC>
	__DEVICE_TAG__ VEC			get(int i, int i1, char x, int i3)const
	{
		VEC	res;
		for (int j = 0;j < dim2;++j) res[j] = (*this)(i,i1,j,i3);
		return res;
	}
	template<class VEC>
	__DEVICE_TAG__ void			get(int i, int i1, char x, int i3, VEC &res)const
	{
		for (int j = 0;j < dim2;++j) res[j] = (*this)(i,i1,j,i3);
	}
	template<class VEC>
	__DEVICE_TAG__ void			set(int i, int i1, char x, int i3, const VEC &v)const
	{
		for (int j = 0;j < dim2;++j) (*this)(i,i1,j,i3) = v[j];
	}
	template<class VEC>
	__DEVICE_TAG__ VEC			get(int i, char x, int i2, int i3)const
	{
		VEC	res;
		for (int j = 0;j < dim1;++j) res[j] = (*this)(i,j,i2,i3);
		return res;
	}
	template<class VEC>
	__DEVICE_TAG__ void			get(int i, char x, int i2, int i3, VEC &res)const
	{
		for (int j = 0;j < dim1;++j) res[j] = (*this)(i,j,i2,i3);
	}
	template<class VEC>
	__DEVICE_TAG__ void			set(int i, char x, int i2, int i3, const VEC &v)const
	{
		for (int j = 0;j < dim1;++j) (*this)(i,j,i2,i3) = v[j];
	}
	__DEVICE_TAG__ t_vec_tml<T,dim3>	getv(int i, int i1, int i2)const { return get<t_vec_tml<T,dim3> >(i, i1, i2); }
	__DEVICE_TAG__ void			getv(int i, int i1, int i2, t_vec_tml<T,dim3> &res)const { get(i, i1, i2, res); }
	__DEVICE_TAG__ void			setv(int i, int i1, int i2, const t_vec_tml<T,dim3> &v)const { set(i, i1, i2, v); }
	__DEVICE_TAG__ t_vec_tml<T,dim2>	getv(int i, int i1, char x, int i3)const { return get<t_vec_tml<T,dim2> >(i, i1, x, i3); }
	__DEVICE_TAG__ void			getv(int i, int i1, char x, int i3, t_vec_tml<T,dim2> &res)const { get(i, i1, x, i3, res); }
	__DEVICE_TAG__ void			setv(int i, int i1, char x, int i3, const t_vec_tml<T,dim2> &v)const { set(i, i1, x, i3, v); }
	__DEVICE_TAG__ t_vec_tml<T,dim1>	getv(int i, char x, int i2, int i3)const { return get<t_vec_tml<T,dim1> >(i, x, i2, i3); }
	__DEVICE_TAG__ void			getv(int i, char x, int i2, int i3, t_vec_tml<T,dim1> &res)const { get(i, x, i2, i3, res); }
	__DEVICE_TAG__ void			setv(int i, char x, int i2, int i3, const t_vec_tml<T,dim1> &v)const { set(i, x, i2, i3, v); }
	//TODO MAT concept
	//TODO in fact we can make several more accessors (like char,int,char and etc); but i doubt they are really needed
	template<class MAT>
	__DEVICE_TAG__ MAT			get2(int i, int i1)const
	{
		MAT	res;
		for (int i2 = 0;i2 < dim2;++i2)
		for (int i3 = 0;i3 < dim3;++i3)
			res(i2,i3) = (*this)(i,i1,i2,i3);
		return res;
	}
        template<class MAT>
	__DEVICE_TAG__ void			get2(int i, int i1, MAT &res)const
	{
		for (int i2 = 0;i2 < dim2;++i2)
		for (int i3 = 0;i3 < dim3;++i3)
			res(i2,i3) = (*this)(i,i1,i2,i3);
	}
	template<class MAT>
	__DEVICE_TAG__ void			set2(int i, int i1, MAT &m)const
	{
		for (int i2 = 0;i2 < dim2;++i2)
		for (int i3 = 0;i3 < dim3;++i3)
			(*this)(i,i1,i2,i3) = m(i2,i3);
	}
};

//ndim is 'spartial' 'dynamic' dimension
//dim1,dim2 - 'small', static dimensions
template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement >
struct t_tensor3_field_tml : public t_tensor3_field_base_tml<T,dim1,dim2,dim3,storage,arrangement>
{
	typedef t_tensor3_field_base_tml<T,dim1,dim2,dim3,storage,arrangement>	t_parent;

	__DEVICE_TAG__ t_tensor3_field_tml() : t_parent() { }
	__DEVICE_TAG__ t_tensor3_field_tml(const t_tensor3_field_tml &tf) : t_parent(tf) { }

	__DEVICE_TAG__ t_tensor3_field_tml	&operator=(const t_tensor3_field_tml &tf) { t_parent::assign(tf); return *this; }

	using 			t_parent::operator();
	using 			t_parent::get;
	using 			t_parent::set;
	//using 			t_parent::host_get;
	using 			t_parent::getv;
	using 			t_parent::setv;
	//using 			t_parent::host_getv;
	using 			t_parent::get2;
	using 			t_parent::set2;

	t_tensor1_field_tml<T,dim1*dim2*dim3,storage,arrangement>	as_tensor1()const;
	t_tensor2_field_tml<T,dim1*dim2,dim3,storage,arrangement>	as_tensor2_merge12()const;
	t_tensor2_field_tml<T,dim1,dim2*dim3,storage,arrangement>	as_tensor2_merge23()const;
	t_tensor4_field_tml<T,1,dim1,dim2,dim3,storage,arrangement>	as_tensor4_f()const;
	t_tensor4_field_tml<T,dim1,dim2,dim3,1,storage,arrangement>	as_tensor4_b()const;
};

template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
struct t_const_tensor3_field_tml : public  t_tensor3_field_base_tml<T,dim1,dim2,dim3,storage,arrangement>
{
	typedef t_tensor3_field_base_tml<T,dim1,dim2,dim3,storage,arrangement>		t_parent;

	__DEVICE_TAG__ t_const_tensor3_field_tml() : t_parent() { }
	//we can create constant field from non-constant
	__DEVICE_TAG__ t_const_tensor3_field_tml(const t_tensor3_field_tml<T,dim1,dim2,dim3,storage> &tf) : t_parent(tf) { }
	__DEVICE_TAG__ t_const_tensor3_field_tml(const t_const_tensor3_field_tml &tf) : t_parent(tf) { }

	__DEVICE_TAG__ t_const_tensor3_field_tml	&operator=(const t_tensor3_field_tml<T,dim1,dim2,dim3,storage> &tf) { t_parent::assign(tf); return *this; }
	__DEVICE_TAG__ t_const_tensor3_field_tml	&operator=(const t_const_tensor3_field_tml &tf) { t_parent::assign(tf); return *this; }

	__DEVICE_TAG__ const  T	&operator()(int i_logic, int i1, int i2, int i3)const { return t_parent::operator()(i_logic, i1, i2, i3); }
	using 			t_parent::get;
	//using 			t_parent::host_get;
	using 			t_parent::getv;
	//using 			t_parent::host_getv;
	using 			t_parent::get2;

	t_const_tensor1_field_tml<T,dim1*dim2*dim3,storage,arrangement>	        as_tensor1()const;
	t_const_tensor2_field_tml<T,dim1*dim2,dim3,storage,arrangement>	        as_tensor2_merge12()const;
	t_const_tensor2_field_tml<T,dim1,dim2*dim3,storage,arrangement>	        as_tensor2_merge23()const;
	t_const_tensor4_field_tml<T,1,dim1,dim2,dim3,storage,arrangement>	as_tensor4_f()const;
	t_const_tensor4_field_tml<T,dim1,dim2,dim3,1,storage,arrangement>	as_tensor4_b()const;
};

//field_storage refers not to view storage but rather to viewed field storage
template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage field_storage, t_tensor_field_arrangement field_arrangement>
struct t_tensor3_field_view_tml : public t_tensor3_field_tml<T,dim1,dim2,dim3,TFS_HOST,field_arrangement>
{
	typedef t_tensor3_field_tml<T,dim1,dim2,dim3,field_storage,field_arrangement>	field_t;
	//see t_tensor0_field_view_tml comments
	typedef field_t							                field_type;
	typedef t_tensor3_field_tml<T,dim1,dim2,dim3,TFS_HOST,field_arrangement>	t_parent;
	const field_t	*f;

	t_tensor3_field_view_tml() : f(NULL) {}
	t_tensor3_field_view_tml(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, int _logic_i0 = 0) : f(NULL)
	{
		init(_f, copy_data, indexing, _logic_i0);
        }
	void	init(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, int _logic_i0 = 0)
	{
		assert(f == NULL);
		f = &_f;
		t_parent::init(f->N, (indexing == TVI_NATIVE?f->logic_i0:_logic_i0));
		if (copy_data) {
			tensor_field_copy_storage_2_host<field_storage>(t_parent::d, f->d, sizeof(T)*t_parent::total_size());
		}
	}
	void	release(bool copy_data = true)
	{
		assert(f != NULL);
		if (copy_data) {
			tensor_field_copy_host_2_storage<field_storage>(f->d, t_parent::d, sizeof(T)*t_parent::total_size());
		}
		this->free();
		f = NULL;
	}

	//ISSUE not sure about this
	int	begin()const { return t_parent::logic_i0; }
	int	end()const { return t_parent::logic_i0+t_parent::size(); }

private:
	t_tensor3_field_view_tml	&operator=(const t_tensor3_field_view_tml &tf_view) { return *this; }	//just to avoid warning
};

//  ------------------------------------------------ TENSOR4 -----------------------------------------------------------

template<int dim1,int dim2,int dim3,int dim4, t_tensor_field_arrangement arrangement>
struct t_tensor4_field_index_base_tml
{
        __DEVICE_TAG__ int calc_idx(int N, int i, int i1, int i2, int i3, int i4)const { return i1*dim2*dim3*dim4*N + i2*dim3*dim4*N + i3*dim4*N + i4*N + i; }
};

template<int dim1,int dim2,int dim3,int dim4>
struct t_tensor4_field_index_base_tml<dim1, dim2, dim3, dim4, TFA_CPU_STYLE>
{
        __DEVICE_TAG__ int calc_idx(int N, int i, int i1, int i2, int i3, int i4)const { return i*dim1*dim2*dim3*dim4 + i1*dim2*dim3*dim4 + i2*dim3*dim4 + i3*dim4 + i4; }
};

//dim1,dim2 - 'small', static dimensions
template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
struct t_tensor4_field_base_tml : public t_tensor4_field_index_base_tml<dim1,dim2,dim3,dim4,arrangement>
{
        typedef t_tensor4_field_index_base_tml<dim1,dim2,dim3,dim4,arrangement>         t_parent;

	//st here stands for static
	static const int					                        st_sz1 = dim1;
	static const int					                        st_sz2 = dim2;
	static const int					                        st_sz3 = dim3;
	static const int					                        st_sz4 = dim4;
	//ISSUE maybe add following indexes (st_sz2, etc) equals 1
	typedef T							                value_type;
	typedef t_tensor4_field_view_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>	view_type;

	int	N;		//'big',dynamic dimension
	T	*d;
	bool	own;    	//whether we should free d-pointed memory when this object dies, for passing in cuda kernels
	int	logic_i0;       //starting logical index for multigpu support (default - 0) (i.e. we enumerate elements of field not from 0 but from logic_i0)

        __DEVICE_TAG__ int size()const { return N; }
	__DEVICE_TAG__ int total_size()const { return size()*dim1*dim2*dim3*dim4; }

	__DEVICE_TAG__ t_tensor4_field_base_tml() : d(NULL) {}
	__DEVICE_TAG__ t_tensor4_field_base_tml(const t_tensor4_field_base_tml &tf) { *this = tf; own = false; }
	__DEVICE_TAG__ ~t_tensor4_field_base_tml()
	{
		#ifndef __CUDA_ARCH__
		if ((d != NULL) && own) free();
		#endif
	}
	void	init(int _N, int _logic_i0 = 0)
	{
		logic_i0 = _logic_i0;
		N = _N;
		own = true;
		tensor_field_malloc<storage>((void**)&d, sizeof(T)*N*dim1*dim2*dim3*dim4);
		assert((d != NULL) && own);
	}
	void	free()
	{
		assert((d != NULL) && own);
		tensor_field_free<storage>(d);
		d = NULL;
		//printf("tensor1:free\n");
	}


protected:
	__DEVICE_TAG__ void			assign(const t_tensor4_field_base_tml &tf) { *this = tf; own = false; }

	__DEVICE_TAG__ T			&operator()(int i_logic,int i1,int i2,int i3,int i4)const { int i = i_logic - logic_i0; return d[t_parent::calc_idx(N, i, i1, i2, i3, i4)]; }
	template<class VEC>
	__DEVICE_TAG__ VEC			get(int i, int i1, int i2, int i3)const
	{
		VEC	res;
		for (int j = 0;j < dim4;++j) res[j] = (*this)(i,i1,i2,i3,j);
		return res;
	}
	template<class VEC>
	__DEVICE_TAG__ void			get(int i, int i1, int i2, int i3, VEC &res)const
	{
		for (int j = 0;j < dim4;++j) res[j] = (*this)(i,i1,i2,i3,j);
	}
	template<class VEC>
	__DEVICE_TAG__ void			set(int i, int i1, int i2, int i3, const VEC &v)const
	{
		for (int j = 0;j < dim4;++j) (*this)(i,i1,i2,i3,j) = v[j];
	}
	template<class VEC>
	__DEVICE_TAG__ VEC			get(int i, int i1, int i2, char x, int i4)const
	{
		VEC	res;
		for (int j = 0;j < dim3;++j) res[j] = (*this)(i,i1,i2,j,i4);
		return res;
	}
	template<class VEC>
	__DEVICE_TAG__ void			get(int i, int i1, int i2, char x, int i4, VEC &res)const
	{
		for (int j = 0;j < dim3;++j) res[j] = (*this)(i,i1,i2,j,i4);
	}
	template<class VEC>
	__DEVICE_TAG__ void			set(int i, int i1, int i2, char x, int i4, const VEC &v)const
	{
		for (int j = 0;j < dim3;++j) (*this)(i,i1,i2,j,i4) = v[j];
	}
	template<class VEC>
	__DEVICE_TAG__ VEC			get(int i, int i1, char x, int i3, int i4)const
	{
		VEC	res;
		for (int j = 0;j < dim2;++j) res[j] = (*this)(i,i1,j,i3,i4);
		return res;
	}
	template<class VEC>
	__DEVICE_TAG__ void			get(int i, int i1, char x, int i3, int i4, VEC &res)const
	{
		for (int j = 0;j < dim2;++j) res[j] = (*this)(i,i1,j,i3,i4);
	}
	template<class VEC>
	__DEVICE_TAG__ void			set(int i, int i1, char x, int i3, int i4, const VEC &v)const
	{
		for (int j = 0;j < dim2;++j) (*this)(i,i1,j,i3,i4) = v[j];
	}
	template<class VEC>
	__DEVICE_TAG__ VEC			get(int i, char x, int i2, int i3, int i4)const
	{
		VEC	res;
		for (int j = 0;j < dim1;++j) res[j] = (*this)(i,j,i2,i3,i4);
		return res;
	}
	template<class VEC>
	__DEVICE_TAG__ void			get(int i, char x, int i2, int i3, int i4, VEC &res)const
	{
		for (int j = 0;j < dim1;++j) res[j] = (*this)(i,j,i2,i3,i4);
	}
	template<class VEC>
	__DEVICE_TAG__ void			set(int i, char x, int i2, int i3, int i4, const VEC &v)const
	{
		for (int j = 0;j < dim1;++j) (*this)(i,j,i2,i3,i4) = v[j];
	}
	__DEVICE_TAG__ t_vec_tml<T,dim4>	getv(int i, int i1, int i2, int i3)const { return get<t_vec_tml<T,dim4> >(i, i1, i2, i3); }
	__DEVICE_TAG__ void			getv(int i, int i1, int i2, int i3, t_vec_tml<T,dim4> &res)const { get(i, i1, i2, i3, res); }
	__DEVICE_TAG__ void			setv(int i, int i1, int i2, int i3, const t_vec_tml<T,dim4> &v)const { set(i, i1, i2, i3, v); }
	__DEVICE_TAG__ t_vec_tml<T,dim3>	getv(int i, int i1, int i2, char x, int i4)const { return get<t_vec_tml<T,dim3> >(i, i1, i2, x, i4); }
	__DEVICE_TAG__ void			getv(int i, int i1, int i2, char x, int i4, t_vec_tml<T,dim3> &res)const { get(i, i1, i2, x, i4, res); }
	__DEVICE_TAG__ void			setv(int i, int i1, int i2, char x, int i4, const t_vec_tml<T,dim3> &v)const { set(i, i1, i2, x, i4, v); }
	__DEVICE_TAG__ t_vec_tml<T,dim2>	getv(int i, int i1, char x, int i3, int i4)const { return get<t_vec_tml<T,dim2> >(i, i1, x, i3, i4); }
	__DEVICE_TAG__ void			getv(int i, int i1, char x, int i3, int i4, t_vec_tml<T,dim2> &res)const { get(i, i1, x, i3, i4, res); }
	__DEVICE_TAG__ void			setv(int i, int i1, char x, int i3, int i4, const t_vec_tml<T,dim2> &v)const { set(i, i1, x, i3, i4, v); }
	__DEVICE_TAG__ t_vec_tml<T,dim1>	getv(int i, char x, int i2, int i3, int i4)const { return get<t_vec_tml<T,dim1> >(i, x, i2, i3, i4); }
	__DEVICE_TAG__ void			getv(int i, char x, int i2, int i3, int i4, t_vec_tml<T,dim1> &res)const { get(i, x, i2, i3, i4, res); }
	__DEVICE_TAG__ void			setv(int i, char x, int i2, int i3, int i4, const t_vec_tml<T,dim1> &v)const { set(i, x, i2, i3, i4, v); }
	//TODO MAT concept
	//TODO in fact we can make several more accessors (like char,int,char and etc); but i doubt they are really needed
	template<class MAT>
	__DEVICE_TAG__ MAT			get2(int i, int i1, int i2)const
	{
		MAT	res;
		for (int i3 = 0;i3 < dim3;++i3)
		for (int i4 = 0;i4 < dim4;++i4)
			res(i3,i4) = (*this)(i,i1,i2,i3,i4);
		return res;
	}
        template<class MAT>
	__DEVICE_TAG__ void			get2(int i, int i1, int i2, MAT &res)const
	{
		for (int i3 = 0;i3 < dim3;++i3)
		for (int i4 = 0;i4 < dim4;++i4)
			res(i3,i4) = (*this)(i,i1,i2,i3,i4);
	}
	template<class MAT>
	__DEVICE_TAG__ void			set2(int i, int i1, int i2, MAT &m)const
	{
		for (int i3 = 0;i3 < dim3;++i3)
		for (int i4 = 0;i4 < dim4;++i4)
			(*this)(i,i1,i2,i3,i4) = m(i3,i4);
	}
};

//ndim is 'spartial' 'dynamic' dimension
//dim1,dim2 - 'small', static dimensions
template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
struct t_tensor4_field_tml : public t_tensor4_field_base_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>
{
	typedef t_tensor4_field_base_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>	t_parent;

	__DEVICE_TAG__ t_tensor4_field_tml() : t_parent() { }
	__DEVICE_TAG__ t_tensor4_field_tml(const t_tensor4_field_tml &tf) : t_parent(tf) { }

	__DEVICE_TAG__ t_tensor4_field_tml	&operator=(const t_tensor4_field_tml &tf) { t_parent::assign(tf); return *this; }

	using 			t_parent::operator();
	using 			t_parent::get;
	using 			t_parent::set;
	//using 			t_parent::host_get;
	using 			t_parent::getv;
	using 			t_parent::setv;
	//using 			t_parent::host_getv;
	using 			t_parent::get2;
	using 			t_parent::set2;

	t_tensor1_field_tml<T,dim1*dim2*dim3*dim4,storage,arrangement>	as_tensor1()const;
	t_tensor3_field_tml<T,dim1*dim2,dim3,dim4,storage,arrangement>	as_tensor3_merge12()const;
	t_tensor3_field_tml<T,dim1,dim2*dim3,dim4,storage,arrangement>	as_tensor3_merge23()const;
	t_tensor3_field_tml<T,dim1,dim2,dim3*dim4,storage,arrangement>	as_tensor3_merge34()const;
};

template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
struct t_const_tensor4_field_tml : public  t_tensor4_field_base_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>
{
	typedef t_tensor4_field_base_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>		t_parent;

	__DEVICE_TAG__ t_const_tensor4_field_tml() : t_parent() { }
	//we can create constant field from non-constant
	__DEVICE_TAG__ t_const_tensor4_field_tml(const t_tensor4_field_tml<T,dim1,dim2,dim3,dim4,storage> &tf) : t_parent(tf) { }
	__DEVICE_TAG__ t_const_tensor4_field_tml(const t_const_tensor4_field_tml &tf) : t_parent(tf) { }

	__DEVICE_TAG__ t_const_tensor4_field_tml	&operator=(const t_tensor4_field_tml<T,dim1,dim2,dim3,dim4,storage> &tf) { t_parent::assign(tf); return *this; }
	__DEVICE_TAG__ t_const_tensor4_field_tml	&operator=(const t_const_tensor4_field_tml &tf) { t_parent::assign(tf); return *this; }

	__DEVICE_TAG__ const  T	&operator()(int i_logic, int i1, int i2, int i3, int i4)const { return t_parent::operator()(i_logic, i1, i2, i3, i4); }
	using 			t_parent::get;
	//using 			t_parent::host_get;
	using 			t_parent::getv;
	//using 			t_parent::host_getv;
	using 			t_parent::get2;

	t_const_tensor1_field_tml<T,dim1*dim2*dim3*dim4,storage,arrangement>	as_tensor1()const;
	t_const_tensor3_field_tml<T,dim1*dim2,dim3,dim4,storage,arrangement>	as_tensor3_merge12()const;
	t_const_tensor3_field_tml<T,dim1,dim2*dim3,dim4,storage,arrangement>	as_tensor3_merge23()const;
	t_const_tensor3_field_tml<T,dim1,dim2,dim3*dim4,storage,arrangement>	as_tensor3_merge34()const;
};

//field_storage refers not to view storage but rather to viewed field storage
template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage field_storage, t_tensor_field_arrangement field_arrangement>
struct t_tensor4_field_view_tml : public t_tensor4_field_tml<T,dim1,dim2,dim3,dim4,TFS_HOST,field_arrangement>
{
	typedef t_tensor4_field_tml<T,dim1,dim2,dim3,dim4,field_storage,field_arrangement>	field_t;
	//see t_tensor0_field_view_tml comments
	typedef field_t								                field_type;
	typedef t_tensor4_field_tml<T,dim1,dim2,dim3,dim4,TFS_HOST,field_arrangement>		t_parent;
	const field_t	*f;

	t_tensor4_field_view_tml() : f(NULL) {}
	t_tensor4_field_view_tml(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, int _logic_i0 = 0) : f(NULL)
	{
		init(_f, copy_data, indexing, _logic_i0);
        }
	void	init(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, int _logic_i0 = 0)
	{
		assert(f == NULL);
		f = &_f;
		t_parent::init(f->N, (indexing == TVI_NATIVE?f->logic_i0:_logic_i0));
		if (copy_data) {
			tensor_field_copy_storage_2_host<field_storage>(t_parent::d, f->d, sizeof(T)*t_parent::total_size());
		}
	}
	void	release(bool copy_data = true)
	{
		assert(f != NULL);
		if (copy_data) {
			tensor_field_copy_host_2_storage<field_storage>(f->d, t_parent::d, sizeof(T)*t_parent::total_size());
		}
		this->free();
		f = NULL;
	}

	//ISSUE not sure about this
	int	begin()const { return t_parent::logic_i0; }
	int	end()const { return t_parent::logic_i0+t_parent::size(); }

private:
	t_tensor4_field_view_tml	&operator=(const t_tensor4_field_view_tml &tf_view) { return *this; }	//just to avoid warning
};

//  ------------------------------------------------ ADAPTERS REALIZATION -----------------------------------------------------------

#define __T_TENSOR_FIELD_TML_ADAPTERS_COPY	\
	res.N = this->N;      			\
	res.d = this->d;                        \
	res.logic_i0 = this->logic_i0;          \
        res.own = false;			\
        return res;

template<typename T,t_tensor_field_storage storage>
t_tensor1_field_tml<T,1,storage>		t_tensor0_field_tml<T,storage>::as_tensor1()const
{
	t_tensor1_field_tml<T,1,storage>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,t_tensor_field_storage storage>
t_tensor2_field_tml<T,1,1,storage>		t_tensor0_field_tml<T,storage>::as_tensor2()const
{
	t_tensor2_field_tml<T,1,1,storage>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,t_tensor_field_storage storage>
t_tensor3_field_tml<T,1,1,1,storage>		t_tensor0_field_tml<T,storage>::as_tensor3()const
{
	t_tensor3_field_tml<T,1,1,1,storage>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,t_tensor_field_storage storage>
t_tensor4_field_tml<T,1,1,1,1,storage>		t_tensor0_field_tml<T,storage>::as_tensor4()const
{
	t_tensor4_field_tml<T,1,1,1,1,storage>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,t_tensor_field_storage storage>
t_const_tensor1_field_tml<T,1,storage>		t_const_tensor0_field_tml<T,storage>::as_tensor1()const
{
	t_const_tensor1_field_tml<T,1,storage>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,t_tensor_field_storage storage>
t_const_tensor2_field_tml<T,1,1,storage>	t_const_tensor0_field_tml<T,storage>::as_tensor2()const
{
	t_const_tensor2_field_tml<T,1,1,storage>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,t_tensor_field_storage storage>
t_const_tensor3_field_tml<T,1,1,1,storage>	t_const_tensor0_field_tml<T,storage>::as_tensor3()const
{
	t_const_tensor3_field_tml<T,1,1,1,storage>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,t_tensor_field_storage storage>
t_const_tensor4_field_tml<T,1,1,1,1,storage>	t_const_tensor0_field_tml<T,storage>::as_tensor4()const
{
	t_const_tensor4_field_tml<T,1,1,1,1,storage>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor2_field_tml<T,1,dim1,storage,arrangement>		t_tensor1_field_tml<T,dim1,storage,arrangement>::as_tensor2_f()const
{
	t_tensor2_field_tml<T,1,dim1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor2_field_tml<T,dim1,1,storage,arrangement>		t_tensor1_field_tml<T,dim1,storage,arrangement>::as_tensor2_b()const
{
	t_tensor2_field_tml<T,dim1,1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor3_field_tml<T,1,1,dim1,storage,arrangement>		t_tensor1_field_tml<T,dim1,storage,arrangement>::as_tensor3_f()const
{
	t_tensor3_field_tml<T,1,1,dim1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor3_field_tml<T,dim1,1,1,storage,arrangement>		t_tensor1_field_tml<T,dim1,storage,arrangement>::as_tensor3_b()const
{
	t_tensor3_field_tml<T,dim1,1,1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor4_field_tml<T,1,1,1,dim1,storage,arrangement>	t_tensor1_field_tml<T,dim1,storage,arrangement>::as_tensor4_f()const
{
	t_tensor4_field_tml<T,1,1,1,dim1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor4_field_tml<T,dim1,1,1,1,storage,arrangement>	t_tensor1_field_tml<T,dim1,storage,arrangement>::as_tensor4_b()const
{
	t_tensor4_field_tml<T,dim1,1,1,1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor2_field_tml<T,1,dim1,storage,arrangement>	t_const_tensor1_field_tml<T,dim1,storage,arrangement>::as_tensor2_f()const
{
	t_const_tensor2_field_tml<T,1,dim1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor2_field_tml<T,dim1,1,storage,arrangement>	t_const_tensor1_field_tml<T,dim1,storage,arrangement>::as_tensor2_b()const
{
	t_const_tensor2_field_tml<T,dim1,1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor3_field_tml<T,1,1,dim1,storage,arrangement>	t_const_tensor1_field_tml<T,dim1,storage,arrangement>::as_tensor3_f()const
{
	t_const_tensor3_field_tml<T,1,1,dim1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor3_field_tml<T,dim1,1,1,storage,arrangement>	t_const_tensor1_field_tml<T,dim1,storage,arrangement>::as_tensor3_b()const
{
	t_const_tensor3_field_tml<T,dim1,1,1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor4_field_tml<T,1,1,1,dim1,storage,arrangement>	t_const_tensor1_field_tml<T,dim1,storage,arrangement>::as_tensor4_f()const
{
	t_const_tensor4_field_tml<T,1,1,1,dim1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor4_field_tml<T,dim1,1,1,1,storage,arrangement>	t_const_tensor1_field_tml<T,dim1,storage,arrangement>::as_tensor4_b()const
{
	t_const_tensor4_field_tml<T,dim1,1,1,1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor1_field_tml<T,dim1*dim2,storage,arrangement>	t_tensor2_field_tml<T,dim1,dim2,storage,arrangement>::as_tensor1()const
{
	t_tensor1_field_tml<T,dim1*dim2,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor3_field_tml<T,1,dim1,dim2,storage,arrangement>	t_tensor2_field_tml<T,dim1,dim2,storage,arrangement>::as_tensor3_f()const
{
	t_tensor3_field_tml<T,1,dim1,dim2,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor3_field_tml<T,dim1,dim2,1,storage,arrangement>	t_tensor2_field_tml<T,dim1,dim2,storage,arrangement>::as_tensor3_b()const
{
	t_tensor3_field_tml<T,dim1,dim2,1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor4_field_tml<T,1,1,dim1,dim2,storage,arrangement>	t_tensor2_field_tml<T,dim1,dim2,storage,arrangement>::as_tensor4_f()const
{
	t_tensor4_field_tml<T,1,1,dim1,dim2,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor4_field_tml<T,dim1,dim2,1,1,storage,arrangement>	t_tensor2_field_tml<T,dim1,dim2,storage,arrangement>::as_tensor4_b()const
{
	t_tensor4_field_tml<T,dim1,dim2,1,1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor1_field_tml<T,dim1*dim2,storage,arrangement>		t_const_tensor2_field_tml<T,dim1,dim2,storage,arrangement>::as_tensor1()const
{
	t_const_tensor1_field_tml<T,dim1*dim2,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor3_field_tml<T,1,dim1,dim2,storage,arrangement>	t_const_tensor2_field_tml<T,dim1,dim2,storage,arrangement>::as_tensor3_f()const
{
	t_const_tensor3_field_tml<T,1,dim1,dim2,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor3_field_tml<T,dim1,dim2,1,storage,arrangement>	t_const_tensor2_field_tml<T,dim1,dim2,storage,arrangement>::as_tensor3_b()const
{
	t_const_tensor3_field_tml<T,dim1,dim2,1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor4_field_tml<T,1,1,dim1,dim2,storage,arrangement>	t_const_tensor2_field_tml<T,dim1,dim2,storage,arrangement>::as_tensor4_f()const
{
	t_const_tensor4_field_tml<T,1,1,dim1,dim2,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor4_field_tml<T,dim1,dim2,1,1,storage,arrangement>	t_const_tensor2_field_tml<T,dim1,dim2,storage,arrangement>::as_tensor4_b()const
{
	t_const_tensor4_field_tml<T,dim1,dim2,1,1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor1_field_tml<T,dim1*dim2*dim3,storage,arrangement>	t_tensor3_field_tml<T,dim1,dim2,dim3,storage,arrangement>::as_tensor1()const
{
	t_tensor1_field_tml<T,dim1*dim2*dim3,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor2_field_tml<T,dim1*dim2,dim3,storage,arrangement>	t_tensor3_field_tml<T,dim1,dim2,dim3,storage,arrangement>::as_tensor2_merge12()const
{
	t_tensor2_field_tml<T,dim1*dim2,dim3,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor2_field_tml<T,dim1,dim2*dim3,storage,arrangement>	t_tensor3_field_tml<T,dim1,dim2,dim3,storage,arrangement>::as_tensor2_merge23()const
{
	t_tensor2_field_tml<T,dim1,dim2*dim3,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor4_field_tml<T,1,dim1,dim2,dim3,storage,arrangement>	t_tensor3_field_tml<T,dim1,dim2,dim3,storage,arrangement>::as_tensor4_f()const
{
	t_tensor4_field_tml<T,1,dim1,dim2,dim3,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor4_field_tml<T,dim1,dim2,dim3,1,storage,arrangement>	t_tensor3_field_tml<T,dim1,dim2,dim3,storage,arrangement>::as_tensor4_b()const
{
	t_tensor4_field_tml<T,dim1,dim2,dim3,1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor1_field_tml<T,dim1*dim2*dim3,storage,arrangement>	t_const_tensor3_field_tml<T,dim1,dim2,dim3,storage,arrangement>::as_tensor1()const
{
	t_const_tensor1_field_tml<T,dim1*dim2*dim3,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor2_field_tml<T,dim1*dim2,dim3,storage,arrangement>	t_const_tensor3_field_tml<T,dim1,dim2,dim3,storage,arrangement>::as_tensor2_merge12()const
{
	t_const_tensor2_field_tml<T,dim1*dim2,dim3,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor2_field_tml<T,dim1,dim2*dim3,storage,arrangement>	t_const_tensor3_field_tml<T,dim1,dim2,dim3,storage,arrangement>::as_tensor2_merge23()const
{
	t_const_tensor2_field_tml<T,dim1,dim2*dim3,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor4_field_tml<T,1,dim1,dim2,dim3,storage,arrangement>	t_const_tensor3_field_tml<T,dim1,dim2,dim3,storage,arrangement>::as_tensor4_f()const
{
	t_const_tensor4_field_tml<T,1,dim1,dim2,dim3,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor4_field_tml<T,dim1,dim2,dim3,1,storage,arrangement>	t_const_tensor3_field_tml<T,dim1,dim2,dim3,storage,arrangement>::as_tensor4_b()const
{
	t_const_tensor4_field_tml<T,dim1,dim2,dim3,1,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor1_field_tml<T,dim1*dim2*dim3*dim4,storage,arrangement>	t_tensor4_field_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>::as_tensor1()const
{
	t_tensor1_field_tml<T,dim1*dim2*dim3*dim4,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor3_field_tml<T,dim1*dim2,dim3,dim4,storage,arrangement>	t_tensor4_field_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>::as_tensor3_merge12()const
{
	t_tensor3_field_tml<T,dim1*dim2,dim3,dim4,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor3_field_tml<T,dim1,dim2*dim3,dim4,storage,arrangement>	t_tensor4_field_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>::as_tensor3_merge23()const
{
	t_tensor3_field_tml<T,dim1,dim2*dim3,dim4,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_tensor3_field_tml<T,dim1,dim2,dim3*dim4,storage,arrangement>	t_tensor4_field_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>::as_tensor3_merge34()const
{
	t_tensor3_field_tml<T,dim1,dim2,dim3*dim4,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor1_field_tml<T,dim1*dim2*dim3*dim4,storage,arrangement>	t_const_tensor4_field_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>::as_tensor1()const
{
	t_const_tensor1_field_tml<T,dim1*dim2*dim3*dim4,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor3_field_tml<T,dim1*dim2,dim3,dim4,storage,arrangement>	t_const_tensor4_field_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>::as_tensor3_merge12()const
{
	t_const_tensor3_field_tml<T,dim1*dim2,dim3,dim4,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor3_field_tml<T,dim1,dim2*dim3,dim4,storage,arrangement>	t_const_tensor4_field_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>::as_tensor3_merge23()const
{
	t_const_tensor3_field_tml<T,dim1,dim2*dim3,dim4,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

template<typename T,int dim1,int dim2,int dim3,int dim4, t_tensor_field_storage storage, t_tensor_field_arrangement arrangement>
t_const_tensor3_field_tml<T,dim1,dim2,dim3*dim4,storage,arrangement>	t_const_tensor4_field_tml<T,dim1,dim2,dim3,dim4,storage,arrangement>::as_tensor3_merge34()const
{
	t_const_tensor3_field_tml<T,dim1,dim2,dim3*dim4,storage,arrangement>	res;
	__T_TENSOR_FIELD_TML_ADAPTERS_COPY
}

#endif
