
#include <iostream>
#include <vecs_mats/t_rect_tml.h>

typedef t_vec_tml<int,3>    idx_t;
typedef t_rect_tml<int,3>   rect_t;

int main()
{
    rect_t  r(idx_t(0,0,0), idx_t(1,1,1));

    std::cout << r.calc_area() << std::endl;

    return 0;
}