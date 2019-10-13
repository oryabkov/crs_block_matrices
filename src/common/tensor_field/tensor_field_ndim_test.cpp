
#include <cstdio>
#include <stdexcept>
#include <vecs_mats/t_vec_tml.h>
#include <tensor_field/t_tensor_field_ndim_tml.h>

#define NDIM    2

typedef t_vec_tml<int,NDIM>                                     t_idx;
typedef t_vec_tml<float,3>                                      t_vec3;
typedef t_tensor1_field_ndim_tml<float,NDIM,3,TFS_HOST>         t_field;
typedef t_field::view_type                                      t_field_view;

int main()
{
        try {

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

        //do stuff
        for (int i = 0;i < 100;++i)
        for (int j = 0;j < 100;++j) {
                f(t_idx(i,j),0) += 1;
                f(t_idx(i,j),1) -= i;
                f(t_idx(i,j),2) -= j;
        }

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