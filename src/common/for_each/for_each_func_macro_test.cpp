
#include <cstdio>
#include "for_each_func_macro.h"

#define __STR2__(x) #x
#define __STR1__(x) __STR2__(x)

struct test
{
        //#pragma message("testtest!!" __STR1__( FOR_EACH_FUNC_NARG(1,1,2,2,3,3,4,4) ))
        //#pragma message("testtest!!" __STR1__( FOR_EACH_FUNC_PARAMS_HELP(test, int, a1, float, b1, int, a2, int, a3) ))
        FOR_EACH_FUNC_PARAMS_HELP(test, int, a1, float, b1, int, a2, int, a3)
};

int main()
{
        test    t(1,1.f,2,3);
        printf("%d %f %d %d\n", t.a1, t.b1, t.a2, t.a3);
        //t.a1 

}