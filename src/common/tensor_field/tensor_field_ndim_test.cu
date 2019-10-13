
#include <stdexcept>
#include <vecs_mats/t_vec_tml.h>
#include <tensor_field/t_tensor_field_ndim_tml.h>
#include <utils/cuda_ownutils.h>

#define NDIM    2

typedef t_vec_tml<int,NDIM>                                     t_idx;
typedef t_vec_tml<float,3>                                      t_vec3;
typedef t_tensor1_field_ndim_tml<float,NDIM,3,TFS_DEVICE>       t_field;
typedef t_field::view_type                                      t_field_view;

__global__ void test_ker(t_field f)
{
        int     i1 = blockIdx.x * blockDim.x + threadIdx.x,
                i2 = blockIdx.y * blockDim.y + threadIdx.y;
        f(t_idx(i1,i2),0) += 1;
        f(t_idx(i1,i2),1) -= i1;
        f(t_idx(i1,i2),2) -= i2;
        
        //t_vec3        v = f.getv<t_vec3>(t_idx(i1,i2));
        t_vec3  v = f.get<t_vec3>(t_idx(i1,i2));
        f.getv(t_idx(i1,i2), v);
        v = f.getv(t_idx(i1,i2));
}

int main()
{
        try {

        if (!InitCUDA(0)) throw std::runtime_error("InitCUDA failed");

        t_field f;
        f.init(t_idx(100,100));

        t_field_view    view(f, false);
        for (int i = 0;i < 100;++i)
        for (int j = 0;j < 100;++j) {
                view(t_idx(i,j), 0) = i;
                view(t_idx(i,j), 1) = i+j;
                view(t_idx(i,j), 2) = i*2+j;
        }
        view.release();

        //call some kernel
        dim3 dimBlock(100,1);
        dim3 dimGrid(1,100);
        test_ker<<<dimGrid, dimBlock>>>(f);

        t_field_view    view2(f, true);
        for (int i = 0;i < 100;++i)
        for (int j = 0;j < 100;++j) {
                printf("%d, %d, %f, %f, %f\n", i, j, view2(t_idx(i,j), 0), view2(t_idx(i,j), 1), view2(t_idx(i,j), 2));
        }
        view2.release();

        return 0;

        } catch(std::exception& e) {

        printf("exception caught: %s\n", e.what());
        return 1;

        }
}