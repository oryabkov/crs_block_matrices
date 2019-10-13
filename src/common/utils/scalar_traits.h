// Copyright © 2016 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCALAR_TRAITS_H__
#define __SCALAR_TRAITS_H__

#include <cmath>
#include <algorithm>

#ifndef __CUDACC__
#define __DEVICE_TAG__
#else
#define __DEVICE_TAG__ __device__ __host__
#endif

namespace simplecfd {

namespace utils {

template<class T>
struct scalar_traits_err
{
	inline __DEVICE_TAG__ static T not_defined() { return T::this_type_is_missing_a_specialization(); };
};

template<class T>
struct scalar_traits
{
	inline __DEVICE_TAG__ static T	zero() { return T(1.f); }
	inline __DEVICE_TAG__ static T	one() { return T(1.f); }
	inline __DEVICE_TAG__ static T	pi() { return T(3.1415926535897932384626433832795f); }
	inline __DEVICE_TAG__ static T	sqrt(const T &x) { return std::sqrt(x); }
	inline __DEVICE_TAG__ static T	abs(const T &x) { return std::abs(x); }
	inline __DEVICE_TAG__ static T	sqr(const T &x) { return x*x; }
	inline __DEVICE_TAG__ static T	iconst(const int &i) { return T(i); }
	inline __DEVICE_TAG__ static T	fconst(const float &f) { return T(f); }
	inline __DEVICE_TAG__ static T	dconst(const double &d) { return T(d); }
	inline __DEVICE_TAG__ static T	max(const T &x,const T &y)
	{
#ifndef __CUDA_ARCH__
		return std::max(x,y);
#else
		return ::max(x,y);	//TODO ???? is it effective
#endif
	}
	inline __DEVICE_TAG__ static T	min(const T &x,const T &y)
	{
#ifndef __CUDA_ARCH__
		return std::min(x,y);
#else
		return ::min(x,y);	//TODO ???? is it effective
#endif
	}
	inline static std::string	name() { return scalar_traits_err<T>::not_defined(); }
};

}

}

#endif
