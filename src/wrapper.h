/* Library API */
#include "mim.h"

struct mim_model *mim_model_create_w(
        size_t parameter_stride,
        const void *parameter_values,
        const size_t images_properties[6],
        void *images_data
);

enum mim_return mim_model_get_w(
        const struct mim_model *model,
        double parameter,
        const size_t image_properties[4],
        void *image_data
);

enum mim_return mim_model_invert_w(
        const struct mim_model *model,
        const size_t observation_properties[4],
        void *observation_data,
        const size_t parameter_properties[4],
        void *parameter_data
);
