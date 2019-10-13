
//#define TEST_HOST
#define TEST_CUDA
//#define TEST_OPENMP

#include <cstdio>
#include <stdexcept>
#include <vecs_mats/t_vec_tml.h>
#include <utils/cuda_ownutils.h>
#include <tensor_field/t_tensor_field_ndim_tml.h>
#include <for_each/for_each_ndim.h>
#ifdef TEST_CUDA
#include <for_each/for_each_ndim_cuda_impl.cuh>
#endif
#ifdef TEST_OPENMP
#include <for_each/for_each_ndim_openmp_impl.h>
#endif
#include <for_each/for_each_storage_types.h>

#ifdef TEST_CUDA
#define FET_TYPE        FET_CUDA
#endif

#ifdef TEST_HOST
#define FET_TYPE        FET_SERIAL_CPU
#endif

#ifdef TEST_OPENMP
#define FET_TYPE        FET_OPENMP
#endif

static const t_tensor_field_storage     TFS_TYPE = t_for_each_storage_type_helper<FET_TYPE>::storage;

//#define DO_RESULTS_OUTPUT
//#define NDIM          2

#define SZ_X    100
//to make it more stressfull
#define SZ_Y    101
#define SZ_Z    102

typedef t_vec_tml<int,1>                                        t_idx1;
typedef t_tensor0_field_ndim_tml<int,1,TFS_TYPE>                t_field0_ndim1;
typedef t_tensor0_field_view_ndim_tml<int,1,TFS_TYPE>           t_field0_ndim1_view;
typedef t_tensor1_field_ndim_tml<int,1,3,TFS_TYPE>              t_field1_ndim1;
typedef t_tensor1_field_view_ndim_tml<int,1,3,TFS_TYPE>         t_field1_ndim1_view;
typedef t_tensor2_field_ndim_tml<int,1,3,4,TFS_TYPE>            t_field2_ndim1;
typedef t_tensor2_field_view_ndim_tml<int,1,3,4,TFS_TYPE>       t_field2_ndim1_view;

typedef t_vec_tml<int,2>                                        t_idx2;
typedef t_tensor0_field_ndim_tml<int,2,TFS_TYPE>                t_field0_ndim2;
typedef t_tensor0_field_view_ndim_tml<int,2,TFS_TYPE>           t_field0_ndim2_view;
typedef t_tensor1_field_ndim_tml<int,2,3,TFS_TYPE>              t_field1_ndim2;
typedef t_tensor1_field_view_ndim_tml<int,2,3,TFS_TYPE>         t_field1_ndim2_view;
typedef t_tensor2_field_ndim_tml<int,2,3,4,TFS_TYPE>            t_field2_ndim2;
typedef t_tensor2_field_view_ndim_tml<int,2,3,4,TFS_TYPE>       t_field2_ndim2_view;

typedef t_vec_tml<int,3>                                        t_idx3;
typedef t_tensor0_field_ndim_tml<int,3,TFS_TYPE>                t_field0_ndim3;
typedef t_tensor0_field_view_ndim_tml<int,3,TFS_TYPE>           t_field0_ndim3_view;
typedef t_tensor1_field_ndim_tml<int,3,3,TFS_TYPE>              t_field1_ndim3;
typedef t_tensor1_field_view_ndim_tml<int,3,3,TFS_TYPE>         t_field1_ndim3_view;
typedef t_tensor2_field_ndim_tml<int,3,3,4,TFS_TYPE>            t_field2_ndim3;
typedef t_tensor2_field_view_ndim_tml<int,3,3,4,TFS_TYPE>       t_field2_ndim3_view;

struct func_test_field0_ndim3
{
        func_test_field0_ndim3(const t_field0_ndim3 &_f) : f(_f) {}
        t_field0_ndim3  f;
        __device__ __host__ void operator()(const t_idx3 &idx)
        {
                f(idx) += 1 - idx[2]*idx[2];
        }
};

bool    test_field0_ndim3()
{
        t_field0_ndim3          f;
        f.init(t_idx3(SZ_X,SZ_Y,SZ_Z));

        t_field0_ndim3_view     view;
        view.init(f, false, TVI_NATIVE);
        for (int i = 0;i < SZ_X;++i)
        for (int j = 0;j < SZ_Y;++j)
        for (int k = 0;k < SZ_Z;++k) {
                view(i, j, k) = i + j - k;
        }
        view.release();

        t_rect_tml<int, 3>              range(t_idx3(0,0,0), t_idx3(SZ_X,SZ_Y,SZ_Z));
        for_each_ndim<FET_TYPE,3>       cuda_foreach;
#ifdef TEST_CUDA
        cuda_foreach.block_size = 128;
#endif
        cuda_foreach(func_test_field0_ndim3(f),range);

        bool    result = true;

        t_field0_ndim3_view     view2;
        view2.init(f, true, TVI_NATIVE);
        for (int i = 0;i < SZ_X;++i)
        for (int j = 0;j < SZ_Y;++j)
        for (int k = 0;k < SZ_Z;++k) {
                if (view2(i, j, k) != i + j - k + 1 - k*k) {
                        printf("test_field0_ndim3: i = %d j = %d k = %d: %d != %d \n", i, j, k, view2(i, j, k), i + j - k + 1 - k*k);
                        result = false;
                }
#ifdef DO_RESULTS_OUTPUT
                printf("%d, %d, %d\n", i, j, view2(i, j, 0));
#endif
        }
        view2.release();

        return result;
}


struct func_test_field1_ndim2
{
        func_test_field1_ndim2(const t_field1_ndim2 &_f) : f(_f) {}
        t_field1_ndim2  f;
        __device__ __host__ void operator()(const t_idx2 &idx)
        {
                f(idx,0) += 1;
                f(idx,1) -= idx[0];
                //f(idx,2) -= idx[1];
                //different type of indexing
                f(idx[0],idx[1],2) -= idx[1];
        }
};

bool    test_field1_ndim2()
{
        t_field1_ndim2          f;
        f.init(t_idx2(SZ_X,SZ_Y));

        t_field1_ndim2_view     view;
        view.init(f, false, TVI_NATIVE);
        for (int i = 0;i < SZ_X;++i)
        for (int j = 0;j < SZ_Y;++j) {
                view(i, j, 0) = i;
                view(i, j, 1) = i+j;
                view(i, j, 2) = i*2+j;
        }
        view.release();

        t_rect_tml<int, 2>              range(t_idx2(0,0), t_idx2(SZ_X,SZ_Y));
        for_each_ndim<FET_TYPE,2>       cuda_foreach;
#ifdef TEST_CUDA
        cuda_foreach.block_size = 128;
#endif
        cuda_foreach(func_test_field1_ndim2(f),range);

        bool    result = true;

        t_field1_ndim2_view     view2;
        view2.init(f, true, TVI_NATIVE);
        for (int i = 0;i < SZ_X;++i)
        for (int j = 0;j < SZ_Y;++j) {
                if (view2(i, j, 0) != i+1) {
                        printf("test_field1_ndim2: i = %d j = %d: %d != %d \n", i, j, view2(i, j, 0), i+1);
                        result = false;
                }
                if (view2(i, j, 1) != i+j-i) {
                        printf("test_field1_ndim2: i = %d j = %d: %d != %d \n", i, j, view2(i, j, 1), i+j-i);
                        result = false;
                }
                if (view2(i, j, 2) != i*2+j-j) {
                        printf("test_field1_ndim2: i = %d j = %d: %d != %d \n", i, j, view2(i, j, 2), i*2+j-j);
                        result = false;
                }
#ifdef DO_RESULTS_OUTPUT
                printf("%d, %d, %d, %d, %d\n", i, j, view2(i, j, 0), view2(i, j, 1), view2(t_idx2(i,j), 2));
#endif
        }
        view2.release();
        
        return result;
}

int main()
{
        try {

#ifdef TEST_CUDA
        if (!InitCUDA(0)) throw std::runtime_error("InitCUDA failed");
#endif
        int err_code = 0;

        if (test_field0_ndim3()) {
                printf("test_field0_ndim3 seems to be OK\n");
        } else {
                printf("test_field0_ndim3 failed\n");
                err_code = 2;
        }

        if (test_field1_ndim2()) {
                printf("test_field1_ndim2 seems to be OK\n");
        } else {
                printf("test_field1_ndim2 failed\n");
                err_code = 2;
        }

        return err_code;
        
        } catch(std::exception& e) {

        printf("exception caught: %s\n", e.what());
        return 1;

        }
}