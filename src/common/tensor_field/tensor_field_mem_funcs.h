#ifndef __TENSOR_FIELD_MALLOC_H__
#define __TENSOR_FIELD_MALLOC_H__

//SCFD_TENSOR_FIELD_CUDA_ENV is now replaces __NVCC__ check (defualt behaviour is still based on __NVCC__ check)
//TODO the problem is that i use '#ifndef SCFD_TENSOR_FIELD_CUDA_ENV' check to check for pure cpp host variant
//but in case we add some other enviroments (like SCFD_TENSOR_FIELD_OPENCL_ENV) it won't work

#include "tensor_field_config.h"
#include <cstdlib>   
#include <cstring>
#include <stdexcept>    
#ifdef SCFD_TENSOR_FIELD_CUDA_ENV
#include <cuda_runtime.h>
#include <utils/cuda_safe_call.h>
#endif
#include "tensor_field_enums.h"

template<t_tensor_field_storage storage>
void	tensor_field_malloc(void** p, size_t size)
{
#ifndef SCFD_TENSOR_FIELD_CUDA_ENV
	if (storage == TFS_DEVICE) {
		//ISSUE hmm seems like not runtime_error
		throw std::runtime_error("tensor_field_malloc: using t_tensor_field_storage TFS_DEVICE in not-cuda code");
	} else {
        	*p = malloc(size);
		if (*p == NULL) throw std::runtime_error("tensor_field_malloc: malloc failed");
 	}
#else
	if (storage == TFS_DEVICE) {
		CUDA_SAFE_CALL(cudaMalloc(p, size));
	} else {
		CUDA_SAFE_CALL(cudaMallocHost(p, size,cudaHostAllocDefault));
	}
#endif
}

template<t_tensor_field_storage storage>
void	tensor_field_free(void* p)
{
#ifndef SCFD_TENSOR_FIELD_CUDA_ENV
	if (storage == TFS_DEVICE) {
		//ISSUE hmm seems like not runtime_error
		throw std::runtime_error("tensor_field_free: using t_tensor_field_storage TFS_DEVICE in not-cuda code");
	} else {
        	free(p);
 	}
#else
	if (storage == TFS_DEVICE) {
		CUDA_SAFE_CALL(cudaFree(p));
	} else {
		CUDA_SAFE_CALL(cudaFreeHost(p));
	}
#endif
}

template<t_tensor_field_storage storage>
void	tensor_field_copy_storage_2_host(void *dst, const void *src, size_t size)
{
#ifndef SCFD_TENSOR_FIELD_CUDA_ENV
	if (storage == TFS_DEVICE) {
		//ISSUE hmm seems like not runtime_error
		throw std::runtime_error("tensor_field_copy_storage_2_host: using t_tensor_field_storage TFS_DEVICE in not-cuda code");
	} else {
		//TODO error handling (how?)
        	memcpy( dst, src, size );
 	}
#else
	if (storage == TFS_DEVICE) {
		CUDA_SAFE_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) );
	} else {
		CUDA_SAFE_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToHost) );
	}
#endif
}

template<t_tensor_field_storage storage>
void	tensor_field_copy_host_2_storage(void *dst, const void *src, size_t size)
{
#ifndef SCFD_TENSOR_FIELD_CUDA_ENV
	if (storage == TFS_DEVICE) {
		//ISSUE hmm seems like not runtime_error
		throw std::runtime_error("tensor_field_copy_host_2_storage: using t_tensor_field_storage TFS_DEVICE in not-cuda code");
	} else {
		//TODO error handling (how?)
        	memcpy( dst, src, size );
 	}
#else
	if (storage == TFS_DEVICE) {
		CUDA_SAFE_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) );
	} else {
		CUDA_SAFE_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToHost) );
	}
#endif
}

#endif
