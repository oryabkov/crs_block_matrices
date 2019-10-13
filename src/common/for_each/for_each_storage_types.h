#ifndef __FOR_EACH_STORAGE_TYPES_H__
#define __FOR_EACH_STORAGE_TYPES_H__

#include "for_each_config.h"
#include <tensor_field/tensor_field_enums.h>
#include "for_each_enums.h"

template<t_for_each_type for_each_type>
struct t_for_each_storage_type_helper
{
};

template<>
struct t_for_each_storage_type_helper<FET_SERIAL_CPU>
{
	static const t_tensor_field_storage storage = TFS_HOST;
};

template<>
struct t_for_each_storage_type_helper<FET_OPENMP>
{
	static const t_tensor_field_storage storage = TFS_HOST;
};

template<>
struct t_for_each_storage_type_helper<FET_CUDA>
{
	static const t_tensor_field_storage storage = TFS_DEVICE;
};

#endif
