#ifndef __T_VEC_TML_H__
#define __T_VEC_TML_H__

#ifndef __CUDACC__
#define __DEVICE_TAG__
#else
#define __DEVICE_TAG__ __device__ __host__
#endif

template<class T,int dim>
struct t_vec_tml
{
	T d[dim];

	__DEVICE_TAG__ t_vec_tml() {}
	__DEVICE_TAG__ t_vec_tml(const T &x0, const T &x1 = T(0.), const T &x2 = T(0.))
	{
		if (dim > 0) d[0] = x0;
		if (dim > 1) d[1] = x1;
		if (dim > 2) d[2] = x2;
	}

	__DEVICE_TAG__ t_vec_tml	operator*(T mul)const
	{
		t_vec_tml	res;
		for (int j = 0;j < dim;++j) res.d[j] = d[j]*mul;
		return res;
	}
	__DEVICE_TAG__ t_vec_tml	operator/(T x)const
	{
		return operator*(T(1.)/x);
	}
	__DEVICE_TAG__ t_vec_tml	operator+(const t_vec_tml &x)const
	{
		t_vec_tml	res;
		for (int j = 0;j < dim;++j) res.d[j] = d[j] + x.d[j];
		return res;
	}
	__DEVICE_TAG__ t_vec_tml	operator-(const t_vec_tml &x)const
	{
		t_vec_tml	res;
		for (int j = 0;j < dim;++j) res.d[j] = d[j] - x.d[j];
		return res;
	}
	__DEVICE_TAG__ T		&operator[](int j) { return d[j]; }
	__DEVICE_TAG__ const T		&operator[](int j)const { return d[j]; }
	__DEVICE_TAG__ T		&operator()(int j) { return d[j]; }
	__DEVICE_TAG__ const T		&operator()(int j)const { return d[j]; }

        __DEVICE_TAG__ T		norm2_sq()const
	{
		T	res(0.);
		for (int j = 0;j < dim;++j) res += d[j]*d[j];
		return res;
	}
	__DEVICE_TAG__ T		norm2()const
	{
		T	res(0.);
		for (int j = 0;j < dim;++j) res += d[j]*d[j];
		return sqrt(res);
	}

	static __DEVICE_TAG__ t_vec_tml	make_zero()
	{
		t_vec_tml	res;
		for (int j = 0;j < dim;++j) res.d[j] = T(0.);
		return res;
	}
	
	template<class T2>
	__DEVICE_TAG__ t_vec_tml &operator=(const t_vec_tml<T2,dim> &v)
	{
		for (int j = 0;j < dim;++j) d[j] = T(v.d[j]);
		return *this;
	}
};

template<class T,int dim>
__DEVICE_TAG__ T	scalar_prod(const t_vec_tml<T,dim> &v1, const t_vec_tml<T,dim> &v2)
{
	T	res(0.);
	for (int j = 0;j < dim;++j) res += v1[j]*v2[j];
	return res;
}

template<class T,int dim>
__DEVICE_TAG__ t_vec_tml<T,dim> vector_prod(const t_vec_tml<T,dim> &v1, const t_vec_tml<T,dim> &v2)
{
	t_vec_tml<T,dim>	res;
	res[0] =   v1[1]*v2[2] - v1[2]*v2[1];
	res[1] = -(v1[0]*v2[2] - v1[2]*v2[0]);
	res[2] =   v1[0]*v2[1] - v1[1]*v2[0];
	return res;
}

typedef t_vec_tml<float,3> t_vec3f;

//temproral solution for matrix ((
template<class T>
__DEVICE_TAG__ T mat33_det(const T mat[3][3])
{
	return	mat[0][0]*(mat[1][1]*mat[2][2]-mat[2][1]*mat[1][2]) -
                mat[1][0]*(mat[0][1]*mat[2][2]-mat[2][1]*mat[0][2]) +
                mat[2][0]*(mat[0][1]*mat[1][2]-mat[1][1]*mat[0][2]);
}

template<class T>
__DEVICE_TAG__ void mat33_inverse(const T mat[3][3],T mat_res[3][3])
{
	T det = mat33_det(mat);
	mat_res[0][0] =  (mat[1][1]*mat[2][2]-mat[2][1]*mat[1][2])/det;
	mat_res[0][1] = -(mat[0][1]*mat[2][2]-mat[2][1]*mat[0][2])/det;
	mat_res[0][2] =  (mat[0][1]*mat[1][2]-mat[1][1]*mat[0][2])/det;

	mat_res[1][0] = -(mat[1][0]*mat[2][2]-mat[2][0]*mat[1][2])/det;
	mat_res[1][1] =  (mat[0][0]*mat[2][2]-mat[2][0]*mat[0][2])/det;
	mat_res[1][2] = -(mat[0][0]*mat[1][2]-mat[1][0]*mat[0][2])/det;

	mat_res[2][0] =  (mat[1][0]*mat[2][1]-mat[2][0]*mat[1][1])/det;
	mat_res[2][1] = -(mat[0][0]*mat[2][1]-mat[2][0]*mat[0][1])/det;
	mat_res[2][2] =  (mat[0][0]*mat[1][1]-mat[1][0]*mat[0][1])/det;
}

template<class T>
__DEVICE_TAG__ t_vec_tml<T,3> prod33(const T mat[3][3],const t_vec_tml<T,3> &v)
{
	t_vec_tml<T,3>	res;
	for (int i = 0;i < 3;++i) {
		res[i] = T(0.);
		for (int j = 0;j < 3;++j) res[i] += mat[i][j]*v[j];
	}
	return res;
}

#endif
