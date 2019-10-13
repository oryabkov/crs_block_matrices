#ifndef __TENSOR_FIELD_CONFIG_H__
#define __TENSOR_FIELD_CONFIG_H__

#define SCFD_TENSOR_FIELD_VERSION_MAJOR  0
#define SCFD_TENSOR_FIELD_VERSION_MINOR 11

//TODO hmm it's hack to overcome problem with my nvcc toolkit 3.2 where __CUDACC__ macro is defined but not __NVCC__
//need to be deleted (figure out from which version there is __NVCC__ macro)
//another reason to think about manual customization
#ifdef __CUDACC__
#ifndef __NVCC__
#define __NVCC__
#endif
#endif

//#define SCFD_TENSOR_FIELD_CPP_ENV
//#define SCFD_TENSOR_FIELD_CUDA_ENV
//#define SCFD_TENSOR_FIELD_OPENCL_ENV

#if !defined(SCFD_TENSOR_FIELD_CPP_ENV) && !defined(SCFD_TENSOR_FIELD_CUDA_ENV) && !defined(SCFD_TENSOR_FIELD_OPENCL_ENV)

#ifdef __NVCC__
#define SCFD_TENSOR_FIELD_CUDA_ENV
#else
#define SCFD_TENSOR_FIELD_CPP_ENV
#endif

#endif

#endif
