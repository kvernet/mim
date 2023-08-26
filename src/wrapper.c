#include "wrapper.h"
/* C standard library */
#include <stdlib.h>

/* Wrapper for numpy arrays. */
struct wrapper {
    struct mim_img pub;
    size_t stride_w;
    size_t stride_h;    
    double *data;
};

static double wrapper_get(
        const struct mim_img *self,
        size_t i, size_t j) {
    struct wrapper *wrapper = (void *)self;
    return wrapper->data[i * wrapper->stride_w + j * wrapper->stride_h];
}

static void wrapper_set(
        struct mim_img *self,
        size_t i, size_t j, double v) {
    struct wrapper *wrapper = (void *)self;
    wrapper->data[i * wrapper->stride_w + j * wrapper->stride_h] = v;
}

static double *wrapper_ref(
        struct mim_img *self,
        size_t i, size_t j) {
    struct wrapper * wrapper = (void *)self;
    return wrapper->data + i * wrapper->stride_w + j * wrapper->stride_h;
}

static void wrapper_destroy(
        struct mim_img **self) {
	if(self == NULL || *self == NULL) return;
	free(*self);
	self = NULL;
}

/* Create an image wrapping an existing numpy array. */
struct mim_img *wrapper_img(
        struct wrapper *wrapper,
        const size_t properties[4],
        void *data) {
    /* Map raw bytes. */
    struct wrapper *w = wrapper;
    
    /* Set public metadata. */
    *(size_t *)&w->pub.width  = properties[0];
    *(size_t *)&w->pub.height = properties[1];
    
    w->pub.get = &wrapper_get;
    w->pub.set = &wrapper_set;
    w->pub.ref = &wrapper_ref;    
    w->pub.destroy = &wrapper_destroy;
    
    /* Set numpy (meta)data. */
    w->stride_w = properties[2] / (sizeof w->data);
    w->stride_h = properties[3] / (sizeof w->data);
    w->data = data;
    
    return &w->pub;
}

struct mim_model *mim_model_create_w(
        size_t parameter_stride,
        const void *parameter_values,
        const size_t images_properties[6],
        void *images_data) {

    /* Wrap parameter vector. */
    size_t size = images_properties[0];
    double parameter[size];
    int i;
    for(i = 0; i < size; i++) {
        parameter[i] = *(double*)(parameter_values + i * parameter_stride);
    }
    
    /* Wrap images collection */
    struct wrapper wrappers[size];
    struct mim_img *images[size];
    const size_t properties[4] = {
        images_properties[1], images_properties[2],
        images_properties[4], images_properties[5]
    };
    
    for(i = 0; i < size; i++) {
        images[i] = wrapper_img(wrappers + i, properties,
                images_data + i * images_properties[3]);
    }
    
    /* Call the library function. */
    struct mim_model *model = mim_model_create(size, parameter, images);    

    return model;
}

enum mim_return mim_model_get_w(
        const struct mim_model *model,
        double parameter,
        const size_t image_properties[4],
        void *image_data) {
    
    struct wrapper wrapper;
    struct mim_img *image = wrapper_img(&wrapper,
            image_properties, image_data);
    
    return mim_model_get(image, model, parameter);
}

enum mim_return mim_model_invert_w(
        const struct mim_model *model,
        const size_t observation_properties[4],
        void *observation_data,
        const size_t filter_properties[4],
        void *filter_data,
        const size_t parameter_properties[4],
        void *parameter_data) {
    
    struct wrapper obs_wrapper, filter_wrapper, img_wrapper;
    const struct mim_img *observation = wrapper_img(
            &obs_wrapper, observation_properties,
            observation_data
    );
    struct mim_img *filter = NULL;
    if(filter_data != NULL) {
        filter = wrapper_img(
                &filter_wrapper, filter_properties,
                filter_data
        );
    }
    struct mim_img *image = wrapper_img(
            &img_wrapper, parameter_properties,
            parameter_data
    );
    
    return mim_model_invert(image, model, observation, filter);
}

enum mim_return mim_model_min_invert_w(
        const struct mim_model *model,
        const size_t observation_properties[4],
        void *observation_data,
        const size_t filter_properties[4],
        void *filter_data,
        const size_t parameter_properties[4],
        void *parameter_data[3],
        const double min_value,
        const double sigma) {
    
    struct wrapper obs_wrapper, filter_wrapper;
    const struct mim_img *observation = wrapper_img(
            &obs_wrapper, observation_properties,
            observation_data
    );
    struct mim_img *filter = NULL;
    if(filter_data != NULL) {
        filter = wrapper_img(
                &filter_wrapper, filter_properties,
                filter_data
        );
    }
    
    struct wrapper parameter_wrapper[3];
    struct mim_img *parameter_images[3];
    for(int i = 0; i < 3; i++) {
        parameter_images[i] = wrapper_img(
                &parameter_wrapper[i], parameter_properties,
                parameter_data[i]);
    }
    
    return mim_model_min_invert(
        parameter_images[0], parameter_images[1],
        parameter_images[2],
        model, observation, filter,
        min_value, sigma);
}
