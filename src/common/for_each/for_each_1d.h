#ifndef __FOR_EACH_1D_H__
#define __FOR_EACH_1D_H__

#include "for_each_config.h"
#include "for_each_enums.h"

//TODO rename for_each_1d to t_for_each_1d_tml; do the same for for_each_ndim; copy this all to tensor_fileds_and_foreach

//T is ordinal type (like int)
//SERIAL_CPU realization is default
template<t_for_each_type type, class T = int>
struct for_each_1d
{
        //FUNC_T concept:
        //TODO
        //copy-constructable
        template<class FUNC_T>
        void operator()(FUNC_T f, int i1, int i2)const
        {
                for (int i = i1;i < i2;++i) {
                        f(i);
                }
        }
        void    wait()const
        {
        }
};

template<class T>
struct for_each_1d<FET_CUDA,T>
{
        //t_rect_tml<T, dim> block_size;        //ISSUE remove from here somehow: we need it just 4 cuda
        int block_size;

        for_each_1d() : block_size(256) {}

        template<class FUNC_T>
        void operator()(FUNC_T f, int i1, int i2)const;
        void wait()const;
};

template<class T>
struct for_each_1d<FET_OPENMP,T>
{
        for_each_1d() : threads_num(-1) {}
        int threads_num;

        template<class FUNC_T>
        void operator()(FUNC_T f, int i1, int i2)const;
        void wait()const;
};

#endif
