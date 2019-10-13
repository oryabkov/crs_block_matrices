#ifndef __TENSOR_FIELD_ENUMS_H__
#define __TENSOR_FIELD_ENUMS_H__

#include "tensor_field_config.h"

enum t_tensor_view_indexing { TVI_CUSTOM, TVI_NATIVE };
enum t_tensor_field_storage { TFS_HOST, TFS_DEVICE };
enum t_tensor_field_arrangement { TFA_CPU_STYLE, TFA_DEVICE_STYLE };

template<t_tensor_field_storage storage_type>
struct t_tensor_field_arrangement_type_helper
{
};

template<>
struct t_tensor_field_arrangement_type_helper<TFS_HOST>
{
        static const t_tensor_field_arrangement arrangement = TFA_CPU_STYLE;
};

template<>
struct t_tensor_field_arrangement_type_helper<TFS_DEVICE>
{
        static const t_tensor_field_arrangement arrangement = TFA_DEVICE_STYLE;
};

#endif
