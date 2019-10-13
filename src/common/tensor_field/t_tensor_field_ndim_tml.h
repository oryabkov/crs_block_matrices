#ifndef __T_TENSOR_FIELD_NDIM_TML_H__
#define __T_TENSOR_FIELD_NDIM_TML_H__

#include "tensor_field_config.h"
#include <cassert>
#include <utils/static_assert.h>
#include <utils/device_tag.h>
#include <vecs_mats/t_vec_tml.h>
#include <vecs_mats/t_rect_tml.h>
#include "tensor_field_enums.h"
#include "tensor_field_mem_funcs.h"

//TODO for non-CUDA case remove all functions specific to CUDA (through macroses, i suppose)
//TODO pass int as 'int' or 'int &' (now it's a big mess)
//TODO seems we have big repetitions here (like get_lin_ind methods from one class to another)
//TODO ordinal type templating (int)
//TODO add host memory allocation specification (like cudaMallocHost vs malloc)

template<typename T,int ndim,t_tensor_field_storage filed_storage = TFS_DEVICE>
struct t_tensor0_field_view_ndim_tml;

//ndim is 'spartial' 'dynamic' dimension
template<typename T,int ndim, t_tensor_field_storage storage>
struct t_tensor0_field_ndim_base_tml
{
	//ISSUE maybe add something like following??
	//static const int						st_sz1 = 1;
	//dyn here stands for 'dynamic'
	static const int						dyn_dims_num = ndim;
	typedef T							value_type;
	typedef	t_vec_tml<int,ndim>					t_idx;
	//*_type nested types are more like something 'official'; t_* are for internal use, but i leave them in public (no hurt from that)
	typedef t_idx							index_type;
	typedef t_tensor0_field_view_ndim_tml<T,ndim,storage>		view_type;

	t_idx	sz;		//'big',dynamic dimensions sizes
	T	*d;
	bool	own;
	t_idx	logic_i0;       //starting logical index for multigpu support (default - 0) (i.e. we enumerate elements of field not from 0 but from logic_i0)

	//ISSUE these 2 called very often; are not they too slow??
        __DEVICE_TAG__ int size()const
        {
		int res = 1;
		for (int j = 0;j < ndim;++j) res *= sz[j];
		return res;
	}
	__DEVICE_TAG__ int total_size()const { return size(); }
	__DEVICE_TAG__ int get_lin_ind(const t_idx &i)const
        {
		int res = 0, mul = 1;
		for (int j = 0;j < ndim;++j) {
			res += mul*i[j];
			mul *= sz[j];
		}
		return res;
	}
	//ISSUE seems like total waste but made 4 uniformity of code
	__DEVICE_TAG__ int get_lin_ind(const int &i_0)const
        {
		STATIC_ASSERT(ndim==1,try_to_use_one_index_access_in_non_one_dim_tensor_field);
		return i_0;
	}
	__DEVICE_TAG__ int get_lin_ind(const int &i_0,const int &i_1)const
        {
		STATIC_ASSERT(ndim==2,try_to_use_two_index_access_in_non_two_dim_tensor_field);
		return i_0 + sz[0]*i_1;
	}
	__DEVICE_TAG__ int get_lin_ind(const int &i_0,const int &i_1,const int &i_2)const
        {
		STATIC_ASSERT(ndim==3,try_to_use_three_index_access_in_non_three_dim_tensor_field);
		return i_0 + sz[0]*i_1 + sz[0]*sz[1]*i_2;
	}

        __DEVICE_TAG__ t_tensor0_field_ndim_base_tml() : d(NULL) {}
	__DEVICE_TAG__ t_tensor0_field_ndim_base_tml(const t_tensor0_field_ndim_base_tml &tf) { *this = tf; own = false; }
	__DEVICE_TAG__ ~t_tensor0_field_ndim_base_tml()
	{
		#ifndef __CUDA_ARCH__
		if ((d != NULL) && own) free();
		#endif
	}
	void	init(t_idx _sz, t_idx _logic_i0 = t_idx::make_zero())
	{
		logic_i0 = _logic_i0;
		sz = _sz;
		own = true;
		tensor_field_malloc<storage>((void**)&d, sizeof(T)*total_size());
		assert((d != NULL) && own);
	}
	void	init(const t_rect_tml<int,ndim> &r)
	{
		init(r.calc_size(), r.i1);
	}
	void	free()
	{
		assert((d != NULL) && own);
		tensor_field_free<storage>(d);
		d = NULL;
		//printf("tensor0:free\n");
	}

protected:
	__DEVICE_TAG__ T			&operator()(const t_idx &i_logic)const
	{
		t_idx i = i_logic - logic_i0;
		return d[get_lin_ind(i)];
	}
	//NOTE don't make assert here becuase it would be done in get_lin_ind method
	//TODO we can use more efficent size() when we now dimension
	__DEVICE_TAG__ T			&operator()(const int &i_0_logic)const
	{ 
		return d[get_lin_ind(i_0_logic - logic_i0[0])]; 
	}
	__DEVICE_TAG__ T			&operator()(const int &i_0_logic,const int &i_1_logic)const 
	{ 
		return d[get_lin_ind(i_0_logic - logic_i0[0],i_1_logic - logic_i0[1])];
	}
	__DEVICE_TAG__ T			&operator()(const int &i_0_logic,const int &i_1_logic,const int &i_2_logic)const
	{
		return d[get_lin_ind(i_0_logic - logic_i0[0],i_1_logic - logic_i0[1],i_2_logic - logic_i0[2])];
	}
};

template<typename T,int ndim, t_tensor_field_storage storage = TFS_DEVICE>
struct t_tensor0_field_ndim_tml : public  t_tensor0_field_ndim_base_tml<T,ndim,storage>
{
	typedef t_tensor0_field_ndim_base_tml<T,ndim,storage>		t_parent;
	typedef	typename t_parent::t_idx				t_idx;

	__DEVICE_TAG__ t_tensor0_field_ndim_tml() : t_parent() { }
	__DEVICE_TAG__ t_tensor0_field_ndim_tml(const t_tensor0_field_ndim_tml &tf) : t_parent(tf) { }

	using 			t_parent::operator();
};

template<typename T,int ndim,t_tensor_field_storage storage = TFS_DEVICE>
struct t_const_tensor0_field_ndim_tml : public  t_tensor0_field_ndim_base_tml<T,ndim,storage>
{
	typedef t_tensor0_field_ndim_base_tml<T,ndim,storage>		t_parent;
	typedef	typename t_parent::t_idx				t_idx;

	__DEVICE_TAG__ t_const_tensor0_field_ndim_tml() : t_parent() { }
	//we can create constant field from non-constant
	__DEVICE_TAG__ t_const_tensor0_field_ndim_tml(const t_tensor0_field_ndim_tml<T,ndim,storage> &tf) : t_parent(tf) { }
	__DEVICE_TAG__ t_const_tensor0_field_ndim_tml(const t_const_tensor0_field_ndim_tml &tf) : t_parent(tf) { }

	__DEVICE_TAG__ const  T	&operator()(const t_idx &i_logic)const { return t_parent::operator()(i_logic); }
	__DEVICE_TAG__ const  T	&operator()(const int &i_0_logic)const { return t_parent::operator()(i_0_logic); }
	__DEVICE_TAG__ const  T	&operator()(const int &i_0_logic,const int &i_1_logic)const { return t_parent::operator()(i_0_logic, i_1_logic); }
	__DEVICE_TAG__ const  T	&operator()(const int &i_0_logic,const int &i_1_logic,const int &i_2_logic)const { return t_parent::operator()(i_0_logic, i_1_logic, i_2_logic); }
};

//filed_storage refers not to view storage but rather to viewed filed storage
template<typename T,int ndim,t_tensor_field_storage filed_storage>
struct t_tensor0_field_view_ndim_tml : public t_tensor0_field_ndim_tml<T,ndim,TFS_HOST>
{
	typedef t_tensor0_field_ndim_tml<T,ndim,filed_storage>		field_t;
	//*_type nested type more like something 'official'; t_* are for internal use, but i leave them in public (no hurt from that)
	//TODO but! why field_t?? i think t_field should be instead
	typedef field_t							field_type;
	typedef t_tensor0_field_ndim_tml<T,ndim,TFS_HOST>		t_parent;
	typedef	t_vec_tml<int,ndim>					t_idx;
	const field_t	*f;

	t_tensor0_field_view_ndim_tml() : f(NULL) {}
	t_tensor0_field_view_ndim_tml(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, t_idx _logic_i0 = t_idx::make_zero()) : f(NULL)
	{
		init(_f, copy_data, indexing, _logic_i0);
        }
	void	init(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, t_idx _logic_i0 = t_idx::make_zero())
	{
		assert(f == NULL);
		f = &_f;
		t_parent::init(f->sz, (indexing == TVI_NATIVE?f->logic_i0:_logic_i0));
		if (copy_data) {
			tensor_field_copy_storage_2_host<filed_storage>(t_parent::d, f->d, sizeof(T)*t_parent::total_size());
		}
	}
	void	release(bool copy_data = true)
	{
		assert(f != NULL);
		if (copy_data) {
			tensor_field_copy_host_2_storage<filed_storage>(f->d, t_parent::d, sizeof(T)*t_parent::total_size());
		}
		this->free();
		f = NULL;
	}
};

template<typename T,int ndim,int dim1, t_tensor_field_storage filed_storage = TFS_DEVICE>
struct t_tensor1_field_view_ndim_tml;

//ndim is 'spartial' 'dynamic' dimension
//dim1 - 'small', static dimension
template<typename T,int ndim,int dim1, t_tensor_field_storage storage>
struct t_tensor1_field_ndim_base_tml
{
	//st here stands for static
	static const int						st_sz1 = dim1;
	//dyn here stands for 'dynamic'
	static const int						dyn_dims_num = ndim;
	typedef T							value_type;
	typedef	t_vec_tml<int,ndim>					t_idx;
	//*_type nested types are more like something 'official'; t_* are for internal use, but i leave them in public (no hurt from that)
	typedef t_idx							index_type;
	typedef t_tensor1_field_view_ndim_tml<T,ndim,dim1,storage>	view_type;

	t_idx	sz;		//'big',dynamic dimensions sizes
	T	*d;
	bool	own;
	t_idx	logic_i0;       //starting logical index for multigpu support (default - 0) (i.e. we enumerate elements of field not from 0 but from logic_i0)

	//ISSUE these 2 called very often; are not they too slow??
        __DEVICE_TAG__ int size()const
        {
		int res = 1;
		for (int j = 0;j < ndim;++j) res *= sz[j];
		return res;
	}
	__DEVICE_TAG__ int total_size()const { return size()*dim1; }
	__DEVICE_TAG__ int get_lin_ind(const t_idx &i)const
        {
		int res = 0, mul = 1;
		for (int j = 0;j < ndim;++j) {
			res += mul*i[j];
			mul *= sz[j];
		}
		return res;
	}
	//ISSUE seems like total waste but made 4 uniformity of code
	__DEVICE_TAG__ int get_lin_ind(const int &i_0)const
        {
		STATIC_ASSERT(ndim==1,try_to_use_one_index_access_in_non_one_dim_tensor_field);
		return i_0;
	}
	__DEVICE_TAG__ int get_lin_ind(const int &i_0,const int &i_1)const
        {
		STATIC_ASSERT(ndim==2,try_to_use_two_index_access_in_non_two_dim_tensor_field);
		return i_0 + sz[0]*i_1;
	}
	__DEVICE_TAG__ int get_lin_ind(const int &i_0,const int &i_1,const int &i_2)const
        {
		STATIC_ASSERT(ndim==3,try_to_use_three_index_access_in_non_three_dim_tensor_field);
		return i_0 + sz[0]*i_1 + sz[0]*sz[1]*i_2;
	}

        __DEVICE_TAG__ t_tensor1_field_ndim_base_tml() : d(NULL) {}
	__DEVICE_TAG__ t_tensor1_field_ndim_base_tml(const t_tensor1_field_ndim_base_tml &tf) { *this = tf; own = false; }
	__DEVICE_TAG__ ~t_tensor1_field_ndim_base_tml()
	{
		#ifndef __CUDA_ARCH__
		if ((d != NULL) && own) free();
		#endif
	}
	void	init(t_idx _sz, t_idx _logic_i0 = t_idx::make_zero())
	{
		logic_i0 = _logic_i0;
		sz = _sz;
		own = true;
		tensor_field_malloc<storage>((void**)&d, sizeof(T)*total_size());
		assert((d != NULL) && own);
	}
	void	init(const t_rect_tml<int,ndim> &r)
	{
		init(r.calc_size(), r.i1);
	}
	void	free()
	{
		assert((d != NULL) && own);
		tensor_field_free<storage>(d);
		d = NULL;
		//printf("tensor1:free\n");
	}

protected:
	__DEVICE_TAG__ T			&operator()(const t_idx &i_logic,int i1)const
	{
		t_idx i = i_logic - logic_i0;
		return d[i1*size() + get_lin_ind(i)];
	}
	//NOTE don't make assert here becuase it would be done in get_lin_ind method
	//TODO we can use more efficent size() when we now dimension
	__DEVICE_TAG__ T			&operator()(const int &i_0_logic,int i1)const
	{ 
		return d[i1*size() + get_lin_ind(i_0_logic - logic_i0[0])]; 
	}
	__DEVICE_TAG__ T			&operator()(const int &i_0_logic,const int &i_1_logic,int i1)const 
	{ 
		return d[i1*size() + get_lin_ind(i_0_logic - logic_i0[0],i_1_logic - logic_i0[1])]; 
	}
	__DEVICE_TAG__ T			&operator()(const int &i_0_logic,const int &i_1_logic,const int &i_2_logic,int i1)const 
	{ 
		return d[i1*size() + get_lin_ind(i_0_logic - logic_i0[0],i_1_logic - logic_i0[1],i_2_logic - logic_i0[2])];
	}
	//VEC concept
	//TODO
	#define __T_TENSOR1_FIELD_NDIM_BASE_TML_ACCESSOR(INDEX_PARAMETERS,INDEX)								\
	template<class VEC>															\
	__DEVICE_TAG__ VEC			get(INDEX_PARAMETERS)const									\
	{                                                                                                                                       \
		VEC	res;															\
		for (int j = 0;j < dim1;++j) res[j] = (*this)(INDEX,j);                                                                     	\
		return res;                                                                                                                     \
	}                                                                                                                                       \
	template<class VEC>                                                                                                                     \
	__DEVICE_TAG__ void			get(INDEX_PARAMETERS, VEC &res)const                                                           	\
	{                                                                                                                                       \
		for (int j = 0;j < dim1;++j) res[j] = (*this)(INDEX,j);                                                       			\
	}                                                                                                                                       \
	template<class VEC>                                                                                                                     \
	__DEVICE_TAG__ void			set(INDEX_PARAMETERS, const VEC &v)const                                                    	\
	{                                                                                                                                       \
		for (int j = 0;j < dim1;++j) (*this)(INDEX,j) = v[j];                                                               		\
	}                                                                                                                                       \
																		\
	template<class VEC>                                                                                                                     \
	void					host_get(INDEX_PARAMETERS, VEC &res)const                                                   	\
	{                                                                                                                                       \
		for (int j = 0;j < dim1;++j) {                                                                                                  \
			tensor_field_copy_storage_2_host<storage>(&(res[j]), &((*this)(INDEX,j)), sizeof(T));                     		\
		}                                                                                                                               \
	}                                                                                                                                       \
	__DEVICE_TAG__ t_vec_tml<T,dim1>	getv(INDEX_PARAMETERS)const { return get<t_vec_tml<T,dim1> >(INDEX); }                     	\
	__DEVICE_TAG__ void			getv(INDEX_PARAMETERS, t_vec_tml<T,dim1> &res)const { get(INDEX, res); }                  	\
	__DEVICE_TAG__ void			setv(INDEX_PARAMETERS, const t_vec_tml<T,dim1> &v)const { setv(INDEX, v); }                	\
	void					host_getv(INDEX_PARAMETERS, t_vec_tml<T,dim1> &res)const { host_get(INDEX, res); }

        //TODO host_get not the best way written (index calc duplication, etc)
        //ISSUE host_get is it safe realisation??

	__T_TENSOR1_FIELD_NDIM_BASE_TML_ACCESSOR(const t_idx &i,i)

        __T_TENSOR1_FIELD_NDIM_BASE_TML_ACCESSOR(const int &i_0,i_0)
	#define __T_TENSOR1_FIELD_NDIM_BASE_TML_INDEX2_COMB(S1,S2) S1, S2
	__T_TENSOR1_FIELD_NDIM_BASE_TML_ACCESSOR(	__T_TENSOR1_FIELD_NDIM_BASE_TML_INDEX2_COMB(const int &i_0,const int &i_1),
							__T_TENSOR1_FIELD_NDIM_BASE_TML_INDEX2_COMB(i_0,i_1))
	#define __T_TENSOR1_FIELD_NDIM_BASE_TML_INDEX3_COMB(S1,S2,S3) S1, S2, S3
	__T_TENSOR1_FIELD_NDIM_BASE_TML_ACCESSOR(	__T_TENSOR1_FIELD_NDIM_BASE_TML_INDEX3_COMB(const int &i_0,const int &i_1,const int &i_2),
							__T_TENSOR1_FIELD_NDIM_BASE_TML_INDEX3_COMB(i_0,i_1,i_2))
};

template<typename T,int ndim,int dim1, t_tensor_field_storage storage = TFS_DEVICE>
struct t_tensor1_field_ndim_tml : public  t_tensor1_field_ndim_base_tml<T,ndim,dim1,storage>
{
	typedef t_tensor1_field_ndim_base_tml<T,ndim,dim1,storage>	t_parent;
	typedef	typename t_parent::t_idx				t_idx;

	__DEVICE_TAG__ t_tensor1_field_ndim_tml() : t_parent() { }
	__DEVICE_TAG__ t_tensor1_field_ndim_tml(const t_tensor1_field_ndim_tml &tf) : t_parent(tf) { }

	using 			t_parent::operator();
	using 			t_parent::get;
	using 			t_parent::set;
	using 			t_parent::host_get;
	using 			t_parent::getv;
	using 			t_parent::setv;
	using 			t_parent::host_getv;
};

template<typename T,int ndim,int dim1, t_tensor_field_storage storage = TFS_DEVICE>
struct t_const_tensor1_field_ndim_tml : public  t_tensor1_field_ndim_base_tml<T,ndim,dim1,storage>
{
	typedef t_tensor1_field_ndim_base_tml<T,ndim,dim1,storage>	t_parent;
	typedef	typename t_parent::t_idx				t_idx;

	__DEVICE_TAG__ t_const_tensor1_field_ndim_tml() : t_parent() { }
	//we can create constant field from non-constant
	__DEVICE_TAG__ t_const_tensor1_field_ndim_tml(const t_tensor1_field_ndim_tml<T,ndim,dim1,storage> &tf) : t_parent(tf) { }
	__DEVICE_TAG__ t_const_tensor1_field_ndim_tml(const t_const_tensor1_field_ndim_tml &tf) : t_parent(tf) { }

	__DEVICE_TAG__ const  T	&operator()(const t_idx &i_logic,int i1)const { return t_parent::operator()(i_logic,i1); }
	__DEVICE_TAG__ const  T	&operator()(const int &i_0_logic,int i1)const { return t_parent::operator()(i_0_logic, i1); }
	__DEVICE_TAG__ const  T	&operator()(const int &i_0_logic,const int &i_1_logic,int i1)const { return t_parent::operator()(i_0_logic, i_1_logic, i1); }
	__DEVICE_TAG__ const  T	&operator()(const int &i_0_logic,const int &i_1_logic,const int &i_2_logic,int i1)const { return t_parent::operator()(i_0_logic, i_1_logic, i_2_logic, i1); }
	using 			t_parent::get;
	using 			t_parent::host_get;
	using 			t_parent::getv;
	using 			t_parent::host_getv;
};

//filed_storage refers not to view storage but rather to viewed filed storage
template<typename T,int ndim,int dim1, t_tensor_field_storage filed_storage>
struct t_tensor1_field_view_ndim_tml : public t_tensor1_field_ndim_tml<T,ndim,dim1,TFS_HOST>
{
	typedef t_tensor1_field_ndim_tml<T,ndim,dim1,filed_storage>	field_t;
	//*_type nested type more like something 'official'; t_* are for internal use, but i leave them in public (no hurt from that)
	//TODO but! why field_t?? i think t_field should be instead
	typedef field_t							field_type;
	typedef t_tensor1_field_ndim_tml<T,ndim,dim1,TFS_HOST>		t_parent;
	typedef	t_vec_tml<int,ndim>					t_idx;
	const field_t	*f;

	t_tensor1_field_view_ndim_tml() : f(NULL) {}
	t_tensor1_field_view_ndim_tml(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, t_idx _logic_i0 = t_idx::make_zero()) : f(NULL)
	{
		init(_f, copy_data, indexing, _logic_i0);
        }
	void	init(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, t_idx _logic_i0 = t_idx::make_zero())
	{
		assert(f == NULL);
		f = &_f;
		t_parent::init(f->sz, (indexing == TVI_NATIVE?f->logic_i0:_logic_i0));
		if (copy_data) {
			tensor_field_copy_storage_2_host<filed_storage>(t_parent::d, f->d, sizeof(T)*t_parent::total_size());
		}
	}
	void	release(bool copy_data = true)
	{
		assert(f != NULL);
		if (copy_data) {
			tensor_field_copy_host_2_storage<filed_storage>(f->d, t_parent::d, sizeof(T)*t_parent::total_size());
		}
		this->free();
		f = NULL;
	}
};

template<typename T,int ndim,int dim1,int dim2, t_tensor_field_storage filed_storage = TFS_DEVICE>
struct t_tensor2_field_view_ndim_tml;

//ndim is 'spartial' 'dynamic' dimension
//dim1,dim2 - 'small', static dimensions
template<typename T,int ndim,int dim1,int dim2, t_tensor_field_storage storage>
struct t_tensor2_field_ndim_base_tml
{
	//st here stands for static
	static const int						st_sz1 = dim1;
	static const int						st_sz2 = dim2;
	//dyn here stands for 'dynamic'
	static const int						dyn_dims_num = ndim;
	typedef T							value_type;
	typedef	t_vec_tml<int,ndim>					t_idx;
	//*_type nested types are more like something 'official'; t_* are for internal use, but i leave them in public (no hurt from that)
	typedef t_idx							index_type;
	typedef t_tensor2_field_view_ndim_tml<T,ndim,dim1,dim2,storage>	view_type;

	t_idx	sz;		//'big',dynamic dimensions sizes
	T	*d;
	bool	own;
	t_idx	logic_i0;       //starting logical index for multigpu support (default - 0) (i.e. we enumerate elements of field not from 0 but from logic_i0)

	//ISSUE these 2 called very often; are not they too slow??
        __DEVICE_TAG__ int size()const
        {
		int res = 1;
		for (int j = 0;j < ndim;++j) res *= sz[j];
		return res;
	}
	__DEVICE_TAG__ int total_size()const { return size()*dim1*dim2; }
	__DEVICE_TAG__ int get_lin_ind(const t_idx &i)const
        {
		int res = 0, mul = 1;
		for (int j = 0;j < ndim;++j) {
			res += mul*i[j];
			mul *= sz[j];
		}
		return res;
	}
	//ISSUE seems like total waste but made 4 uniformity of code
	__DEVICE_TAG__ int get_lin_ind(const int &i_0)const
        {
		STATIC_ASSERT(ndim==1,try_to_use_one_index_access_in_non_one_dim_tensor_field);
		return i_0;
	}
	__DEVICE_TAG__ int get_lin_ind(const int &i_0,const int &i_1)const
        {
		STATIC_ASSERT(ndim==2,try_to_use_two_index_access_in_non_two_dim_tensor_field);
		return i_0 + sz[0]*i_1;
	}
	__DEVICE_TAG__ int get_lin_ind(const int &i_0,const int &i_1,const int &i_2)const
        {
		STATIC_ASSERT(ndim==3,try_to_use_three_index_access_in_non_three_dim_tensor_field);
		return i_0 + sz[0]*i_1 + sz[0]*sz[1]*i_2;
	}


        __DEVICE_TAG__ t_tensor2_field_ndim_base_tml() : d(NULL) {}
	__DEVICE_TAG__ t_tensor2_field_ndim_base_tml(const t_tensor2_field_ndim_base_tml &tf) { *this = tf; own = false; }
	__DEVICE_TAG__ ~t_tensor2_field_ndim_base_tml()
	{
		#ifndef __CUDA_ARCH__
		if ((d != NULL) && own) free();
		#endif
	}
	void	init(t_idx _sz, t_idx _logic_i0 = t_idx::make_zero())
	{
		logic_i0 = _logic_i0;
		sz = _sz;
		own = true;
		tensor_field_malloc<storage>((void**)&d, sizeof(T)*total_size());
		assert((d != NULL) && own);
	}
	void	init(const t_rect_tml<int,ndim> &r)
	{
		init(r.calc_size(), r.i1);
	}
	void	free()
	{
		assert((d != NULL) && own);
		tensor_field_free<storage>(d);
		d = NULL;
	}

protected:
	__DEVICE_TAG__ T			&operator()(const t_idx &i_logic, int i1, int i2)const { t_idx i = i_logic - logic_i0; return d[i1*dim2*size() + i2*size() + get_lin_ind(i)]; }
	//NOTE don't make assert here becuase it would be done in get_lin_ind method
	//TODO we can use more efficent size() when we now dimension
	__DEVICE_TAG__ T			&operator()(const int &i_0_logic,int i1, int i2)const
	{
		return d[i1*dim2*size() + i2*size() + get_lin_ind(i_0_logic - logic_i0[0])];
	}
	__DEVICE_TAG__ T			&operator()(const int &i_0_logic,const int &i_1_logic,int i1, int i2)const
	{
		return d[i1*dim2*size() + i2*size() + get_lin_ind(i_0_logic - logic_i0[0],i_1_logic - logic_i0[1])];
	}
	__DEVICE_TAG__ T			&operator()(const int &i_0_logic,const int &i_1_logic,const int &i_2_logic,int i1, int i2)const
	{
		return d[i1*dim2*size() + i2*size() + get_lin_ind(i_0_logic - logic_i0[0],i_1_logic - logic_i0[1],i_2_logic - logic_i0[2])];
	}
	//VEC,MAT concept
	//TODO
	#define __T_TENSOR2_FIELD_NDIM_BASE_TML_ACCESSOR(INDEX_PARAMETERS,INDEX)									\
	template<class VEC>																\
	__DEVICE_TAG__ VEC			get(INDEX_PARAMETERS, int i1)const									\
	{																		\
		VEC	res;																\
		for (int j = 0;j < dim2;++j) res[j] = (*this)(INDEX,i1,j);										\
		return res;																\
	}																		\
	template<class VEC>																\
	__DEVICE_TAG__ void			get(INDEX_PARAMETERS, int i1, VEC &res)const								\
	{																		\
		for (int j = 0;j < dim2;++j) res[j] = (*this)(INDEX,i1,j);										\
	}																		\
	template<class VEC>																\
	__DEVICE_TAG__ void			set(INDEX_PARAMETERS, int i1, const VEC &v)const							\
	{																		\
		for (int j = 0;j < dim2;++j) (*this)(INDEX,i1,j) = v[j];										\
	}																		\
	template<class VEC>																\
	__DEVICE_TAG__ VEC			get(INDEX_PARAMETERS, char x, int i2)const								\
	{																		\
		VEC	res;																\
		for (int j = 0;j < dim1;++j) res[j] = (*this)(INDEX,j,i2);										\
		return res;																\
	}																		\
	template<class VEC>																\
	__DEVICE_TAG__ void			get(INDEX_PARAMETERS, char x, int i2, VEC &res)const							\
	{																		\
		for (int j = 0;j < dim1;++j) res[j] = (*this)(INDEX,j,i2);										\
	}																		\
	template<class VEC>																\
	__DEVICE_TAG__ void			set(INDEX_PARAMETERS, char x, int i2, const VEC &v)const						\
	{																		\
		for (int j = 0;j < dim1;++j) (*this)(INDEX,j,i2) = v[j];										\
	}																		\
	__DEVICE_TAG__ t_vec_tml<T,dim2>	getv(INDEX_PARAMETERS, int i1)const { return get<t_vec_tml<T,dim2> >(INDEX, i1); }			\
	__DEVICE_TAG__ void			getv(INDEX_PARAMETERS, int i1, t_vec_tml<T,dim2> &res)const { get(INDEX, i1, res); }			\
	__DEVICE_TAG__ void			setv(INDEX_PARAMETERS, int i1, const t_vec_tml<T,dim2> &v)const { setv(INDEX, i1, v); }			\
	__DEVICE_TAG__ t_vec_tml<T,dim1>	getv(INDEX_PARAMETERS, char x, int i2)const { return get<t_vec_tml<T,dim1> >(INDEX, x, i2); }		\
	__DEVICE_TAG__ void			getv(INDEX_PARAMETERS, char x, int i2, t_vec_tml<T,dim1> &res)const { get(INDEX, x, i2, res); }		\
	__DEVICE_TAG__ void			setv(INDEX_PARAMETERS, char x, int i2, const t_vec_tml<T,dim1> &v)const { setv(INDEX, x, i2, v); }	\
	template<class MAT>																\
	__DEVICE_TAG__ MAT			get2(INDEX_PARAMETERS)const										\
	{																		\
		MAT	res;																\
		for (int i1 = 0;i1 < dim1;++i1)														\
		for (int i2 = 0;i2 < dim2;++i2)														\
			res(i1,i2) = (*this)(INDEX,i1,i2);												\
		return res;																\
	}																		\
        template<class MAT>																\
	__DEVICE_TAG__ void			get2(INDEX_PARAMETERS, MAT &res)const									\
	{																		\
		for (int i1 = 0;i1 < dim1;++i1)														\
		for (int i2 = 0;i2 < dim2;++i2)														\
			res(i1,i2) = (*this)(INDEX,i1,i2);												\
	}																		\
	template<class MAT>																\
	__DEVICE_TAG__ void			set2(INDEX_PARAMETERS, MAT &m)const									\
	{																		\
		for (int i1 = 0;i1 < dim1;++i1)														\
		for (int i2 = 0;i2 < dim2;++i2)														\
			(*this)(INDEX,i1,i2) = m(i1,i2);												\
	}
	
	__T_TENSOR2_FIELD_NDIM_BASE_TML_ACCESSOR(const t_idx &i,i)

        __T_TENSOR2_FIELD_NDIM_BASE_TML_ACCESSOR(const int &i_0,i_0)
	#define __T_TENSOR2_FIELD_NDIM_BASE_TML_INDEX2_COMB(S1,S2) S1, S2
	__T_TENSOR2_FIELD_NDIM_BASE_TML_ACCESSOR(	__T_TENSOR2_FIELD_NDIM_BASE_TML_INDEX2_COMB(const int &i_0,const int &i_1),
							__T_TENSOR2_FIELD_NDIM_BASE_TML_INDEX2_COMB(i_0,i_1))
	#define __T_TENSOR2_FIELD_NDIM_BASE_TML_INDEX3_COMB(S1,S2,S3) S1, S2, S3
	__T_TENSOR2_FIELD_NDIM_BASE_TML_ACCESSOR(	__T_TENSOR2_FIELD_NDIM_BASE_TML_INDEX3_COMB(const int &i_0,const int &i_1,const int &i_2),
							__T_TENSOR2_FIELD_NDIM_BASE_TML_INDEX3_COMB(i_0,i_1,i_2))

};

//ndim is 'spartial' 'dynamic' dimension
//dim1,dim2 - 'small', static dimensions
template<typename T,int ndim,int dim1,int dim2, t_tensor_field_storage storage = TFS_DEVICE>
struct t_tensor2_field_ndim_tml : public t_tensor2_field_ndim_base_tml<T,ndim,dim1,dim2,storage>
{
	typedef t_tensor2_field_ndim_base_tml<T,ndim,dim1,dim2,storage>	t_parent;
	typedef	typename t_parent::t_idx				t_idx;

	__DEVICE_TAG__ t_tensor2_field_ndim_tml() : t_parent() { }
	__DEVICE_TAG__ t_tensor2_field_ndim_tml(const t_tensor2_field_ndim_tml &tf) : t_parent(tf) { }

	using 			t_parent::operator();
	using 			t_parent::get;
	using 			t_parent::set;
	//using 			t_parent::host_get;
	using 			t_parent::getv;
	using 			t_parent::setv;
	//using 			t_parent::host_getv;
	using 			t_parent::get2;
	using 			t_parent::set2;
};

template<typename T,int ndim,int dim1,int dim2, t_tensor_field_storage storage = TFS_DEVICE>
struct t_const_tensor2_field_ndim_tml : public  t_tensor2_field_ndim_base_tml<T,ndim,dim1,dim2,storage>
{
	typedef t_tensor2_field_ndim_base_tml<T,ndim,dim1,dim2,storage>	t_parent;
	typedef	typename t_parent::t_idx				t_idx;

	__DEVICE_TAG__ t_const_tensor2_field_ndim_tml() : t_parent() { }
	//we can create constant field from non-constant
	__DEVICE_TAG__ t_const_tensor2_field_ndim_tml(const t_tensor2_field_ndim_tml<T,ndim,dim1,dim2,storage> &tf) : t_parent(tf) { }
	__DEVICE_TAG__ t_const_tensor2_field_ndim_tml(const t_const_tensor2_field_ndim_tml &tf) : t_parent(tf) { }

	__DEVICE_TAG__ const  T	&operator()(const t_idx &i_logic, int i1, int i2)const { return t_parent::operator()(i_logic, i1, i2); }
	__DEVICE_TAG__ const  T	&operator()(const int &i_0_logic,int i1, int i2)const { return t_parent::operator()(i_0_logic, i1, i2); }
	__DEVICE_TAG__ const  T	&operator()(const int &i_0_logic,const int &i_1_logic,int i1, int i2)const { return t_parent::operator()(i_0_logic, i_1_logic, i1, i2); }
	__DEVICE_TAG__ const  T	&operator()(const int &i_0_logic,const int &i_1_logic,const int &i_2_logic,int i1, int i2)const { return t_parent::operator()(i_0_logic, i_1_logic, i_2_logic, i1, i2); }
	using 			t_parent::get;
	//using 			t_parent::host_get;
	using 			t_parent::getv;
	//using 			t_parent::host_getv;
	using 			t_parent::get2;
};

//filed_storage refers not to view storage but rather to viewed filed storage
template<typename T,int ndim,int dim1,int dim2, t_tensor_field_storage filed_storage>
struct t_tensor2_field_view_ndim_tml : public t_tensor2_field_ndim_tml<T,ndim,dim1,dim2,TFS_HOST>
{
	typedef t_tensor2_field_ndim_tml<T,ndim,dim1,dim2,filed_storage>	field_t;
	//*_type nested type more like something 'official'; t_* are for internal use, but i leave them in public (no hurt from that)
	//TODO but! why field_t?? i think t_field should be instead
	typedef field_t								field_type;
	typedef t_tensor2_field_ndim_tml<T,ndim,dim1,dim2,TFS_HOST>		t_parent;
	typedef	t_vec_tml<int,ndim>						t_idx;
	const field_t	*f;

	t_tensor2_field_view_ndim_tml() : f(NULL) {}
	t_tensor2_field_view_ndim_tml(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, t_idx _logic_i0 = t_idx::make_zero()) : f(NULL)
	{
		init(_f, copy_data, indexing, _logic_i0);
        }
	void	init(const field_t &_f, bool copy_data = true, t_tensor_view_indexing indexing = TVI_NATIVE, t_idx _logic_i0 = t_idx::make_zero())
	{
		assert(f == NULL);
		f = &_f;
		t_parent::init(f->sz, (indexing == TVI_NATIVE?f->logic_i0:_logic_i0));
		if (copy_data) {
			tensor_field_copy_storage_2_host<filed_storage>(t_parent::d, f->d, sizeof(T)*t_parent::total_size());
		}
	}
	void	release(bool copy_data = true)
	{
		assert(f != NULL);
		if (copy_data) {
			tensor_field_copy_host_2_storage<filed_storage>(f->d, t_parent::d, sizeof(T)*t_parent::total_size());
		}
		this->free();
		f = NULL;
	}
};

#endif
