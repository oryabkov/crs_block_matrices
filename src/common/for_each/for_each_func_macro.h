#ifndef __FOR_EACH_FUNC_MACRO_H__
#define __FOR_EACH_FUNC_MACRO_H__

#include "for_each_config.h"

//TODO not working for MSVC 2008
//i suppose we can use boost instead?
//NOTE however this variant is totally standart-compliant

#define FOR_EACH_FUNC_CONCATENATE(arg1, arg2)   FOR_EACH_FUNC_CONCATENATE1(arg1, arg2)
#define FOR_EACH_FUNC_CONCATENATE1(arg1, arg2)  FOR_EACH_FUNC_CONCATENATE2(arg1, arg2)
#define FOR_EACH_FUNC_CONCATENATE2(arg1, arg2)  arg1##arg2

#define FOR_EACH_FUNC_NARG(...) FOR_EACH_FUNC_NARG_(__VA_ARGS__, FOR_EACH_FUNC_RSEQ_N())
#define FOR_EACH_FUNC_NARG_(...) FOR_EACH_FUNC_ARG_N(__VA_ARGS__) 
#define FOR_EACH_FUNC_ARG_N(_11, _12, _21, _22, _31, _32, _41, _42, _51, _52, _61, _62, _71, _72, _81, _82, _91, _92, _101, _102, _111, _112, _121, _122, N, ...) N 
#define FOR_EACH_FUNC_RSEQ_N() 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0

#define FOR_EACH_FUNC_PARAMS_DEF_1(par_type, par_name, ...) par_type par_name;
#define FOR_EACH_FUNC_PARAMS_DEF_2(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  FOR_EACH_FUNC_PARAMS_DEF_1(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_DEF_3(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  FOR_EACH_FUNC_PARAMS_DEF_2(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_DEF_4(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  FOR_EACH_FUNC_PARAMS_DEF_3(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_DEF_5(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  FOR_EACH_FUNC_PARAMS_DEF_4(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_DEF_6(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  FOR_EACH_FUNC_PARAMS_DEF_5(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_DEF_7(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  FOR_EACH_FUNC_PARAMS_DEF_6(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_DEF_8(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  FOR_EACH_FUNC_PARAMS_DEF_7(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_DEF_9(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  FOR_EACH_FUNC_PARAMS_DEF_8(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_DEF_10(par_type, par_name, ...)                            \
  par_type par_name;                                                                    \
  FOR_EACH_FUNC_PARAMS_DEF_9(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_DEF_11(par_type, par_name, ...)                            \
  par_type par_name;                                                                    \
  FOR_EACH_FUNC_PARAMS_DEF_10(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_DEF_12(par_type, par_name, ...)                            \
  par_type par_name;                                                                    \
  FOR_EACH_FUNC_PARAMS_DEF_11(__VA_ARGS__)

#define FOR_EACH_FUNC_PARAMS_DEF_(N, ...) FOR_EACH_FUNC_CONCATENATE(FOR_EACH_FUNC_PARAMS_DEF_, N)(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_DEF(...) FOR_EACH_FUNC_PARAMS_DEF_(FOR_EACH_FUNC_NARG(__VA_ARGS__), __VA_ARGS__)

#define FOR_EACH_FUNC_PARAMS_LIST_1(par_type, par_name, ...) par_type FOR_EACH_FUNC_CONCATENATE(_,par_name)
#define FOR_EACH_FUNC_PARAMS_LIST_2(par_type, par_name, ...)                                                            \
  par_type FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  FOR_EACH_FUNC_PARAMS_LIST_1(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_LIST_3(par_type, par_name, ...)                                                            \
  par_type FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  FOR_EACH_FUNC_PARAMS_LIST_2(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_LIST_4(par_type, par_name, ...)                                                            \
  par_type FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  FOR_EACH_FUNC_PARAMS_LIST_3(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_LIST_5(par_type, par_name, ...)                                                            \
  par_type FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  FOR_EACH_FUNC_PARAMS_LIST_4(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_LIST_6(par_type, par_name, ...)                                                            \
  par_type FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  FOR_EACH_FUNC_PARAMS_LIST_5(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_LIST_7(par_type, par_name, ...)                                                            \
  par_type FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  FOR_EACH_FUNC_PARAMS_LIST_6(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_LIST_8(par_type, par_name, ...)                                                            \
  par_type FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  FOR_EACH_FUNC_PARAMS_LIST_7(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_LIST_9(par_type, par_name, ...)                                                            \
  par_type FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  FOR_EACH_FUNC_PARAMS_LIST_8(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_LIST_10(par_type, par_name, ...)                                                           \
  par_type FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  FOR_EACH_FUNC_PARAMS_LIST_9(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_LIST_11(par_type, par_name, ...)                                                           \
  par_type FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  FOR_EACH_FUNC_PARAMS_LIST_10(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_LIST_12(par_type, par_name, ...)                                                           \
  par_type FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  FOR_EACH_FUNC_PARAMS_LIST_11(__VA_ARGS__)

#define FOR_EACH_FUNC_PARAMS_LIST_(N, ...) FOR_EACH_FUNC_CONCATENATE(FOR_EACH_FUNC_PARAMS_LIST_, N)(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_LIST(...) FOR_EACH_FUNC_PARAMS_LIST_(FOR_EACH_FUNC_NARG(__VA_ARGS__), __VA_ARGS__)

#define FOR_EACH_FUNC_PARAMS_CC_1(par_type, par_name, ...) par_name(FOR_EACH_FUNC_CONCATENATE(_,par_name))
#define FOR_EACH_FUNC_PARAMS_CC_2(par_type, par_name, ...)                                                              \
  par_name(FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  FOR_EACH_FUNC_PARAMS_CC_1(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_CC_3(par_type, par_name, ...)                                                              \
  par_name(FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  FOR_EACH_FUNC_PARAMS_CC_2(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_CC_4(par_type, par_name, ...)                                                              \
  par_name(FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  FOR_EACH_FUNC_PARAMS_CC_3(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_CC_5(par_type, par_name, ...)                                                              \
  par_name(FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  FOR_EACH_FUNC_PARAMS_CC_4(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_CC_6(par_type, par_name, ...)                                                              \
  par_name(FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  FOR_EACH_FUNC_PARAMS_CC_5(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_CC_7(par_type, par_name, ...)                                                              \
  par_name(FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  FOR_EACH_FUNC_PARAMS_CC_6(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_CC_8(par_type, par_name, ...)                                                              \
  par_name(FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  FOR_EACH_FUNC_PARAMS_CC_7(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_CC_9(par_type, par_name, ...)                                                              \
  par_name(FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  FOR_EACH_FUNC_PARAMS_CC_8(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_CC_10(par_type, par_name, ...)                                                             \
  par_name(FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  FOR_EACH_FUNC_PARAMS_CC_9(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_CC_11(par_type, par_name, ...)                                                             \
  par_name(FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  FOR_EACH_FUNC_PARAMS_CC_10(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_CC_12(par_type, par_name, ...)                                                             \
  par_name(FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  FOR_EACH_FUNC_PARAMS_CC_11(__VA_ARGS__)

#define FOR_EACH_FUNC_PARAMS_CC_(N, ...) FOR_EACH_FUNC_CONCATENATE(FOR_EACH_FUNC_PARAMS_CC_, N)(__VA_ARGS__)
#define FOR_EACH_FUNC_PARAMS_CC(...) FOR_EACH_FUNC_PARAMS_CC_(FOR_EACH_FUNC_NARG(__VA_ARGS__), __VA_ARGS__)

#define FOR_EACH_FUNC_PARAMS_HELP(func_name, ...)                                                                       \
        FOR_EACH_FUNC_PARAMS_DEF(__VA_ARGS__)                                                                           \
        func_name(FOR_EACH_FUNC_PARAMS_LIST(__VA_ARGS__)) : FOR_EACH_FUNC_PARAMS_CC(__VA_ARGS__) {}     

#endif
