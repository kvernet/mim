#include "mim.h"

#include <stdio.h>
#include <stdlib.h>

double model_par(const double par);

void fill_img(struct mim_img **img, const double par,
        struct mim_prng *prng);

struct mim_model *create_model(
        const size_t size, const double *parameter,
        const size_t width, const size_t height,
        struct mim_prng *prng);

int main() {
    
    const size_t width = 200;
    const size_t height = 100;
    const double par_min = 1.0;
    const double par_max = 3.0;
    const size_t npar = 20;
    double parameter[npar];
    const double par_delta = (par_max - par_min) / (npar - 1);
    for(size_t i = 0; i < npar; i++) {
        parameter[i] = par_min + i * par_delta;
    }
    const double par = 1.8;
    const double min_value = 800;
    const double sigma = 1.;    /* gaussian sigma for weights */
    
    // init the prng
    struct mim_prng *prng = mim_prng_init(0);
    
    // create the model
    struct mim_model *model = create_model(
            npar, parameter,
            width, height, NULL);
    // create the observation
    struct mim_img *observation = mim_img_empty(width, height);
    fill_img(&observation, model_par(par), prng);
    
    // create the filter
    struct mim_img *filter = mim_img_empty(width, height);
    fill_img(&filter, 1, NULL);
    
    // return images
    struct mim_img *image = mim_img_zeros(width, height);
    struct mim_img *bin_image = mim_img_zeros(width, height);
    struct mim_img *value_image = mim_img_zeros(width, height);
    
    enum mim_return mrc = mim_model_min_invert(
            image, bin_image, value_image,
            model, observation, filter,
            min_value, sigma);
    
    if(mrc == MIM_FAILURE) {
        fputs("error(mim) - an error occured.\n", stderr);
        return EXIT_FAILURE;
    }
    
    printf("(  i,   j) [obs]     (min,      sum)       (bins)   *** par    ***\n");
    size_t i, j;
    for(i = 0; i < width; i++) {
        for(j = 0; j < height; j++) {
            printf("(%3ld, %3ld) [%3.4f] (%3.4f, %3.4f) (%3.4f) *** %3.4f ***\n",
                    i, j, 
                    observation->get(observation, i, j),
                    min_value,
                    value_image->get(value_image, i, j),
                    bin_image->get(bin_image, i, j),
                    image->get(image, i, j)
            );
        }
    }
    printf("(  i,   j) [obs]     (min,      sum)       (bins)   *** par    ***\n");
    
    // free allocated memory
    image->destroy(&image);
    bin_image->destroy(&bin_image);
    value_image->destroy(&value_image);
    
    model->destroy(&model);
    observation->destroy(&observation);
    filter->destroy(&filter);
    
    prng->destroy(&prng);
    
    if(mrc == MIM_FAILURE) {
        fputs("error(mim) - an error occured.\n", stderr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

double model_par(const double par) {
    return 100 - 3.5*par*par;
}

void fill_img(struct mim_img **img, const double value,
        struct mim_prng *prng) {
    const size_t width = (*img)->width;
    const size_t height = (*img)->height;
    
    size_t i, j;
    for(i = 0; i < width; i++) {
        for(j = 0; j < height; j++) {
            double v = value;
            if(prng != NULL) v = prng->poisson(prng, value);
            (*img)->set(*img, i, j, v);
        }
    }
}

struct mim_model *create_model(
        const size_t size, const double *parameter,
        const size_t width, const size_t height,
        struct mim_prng *prng) {
    if(size <= 0) return NULL;
    
    struct mim_img *images[size];
    for(size_t i = 0; i < size; i++) {
        images[i] = mim_img_empty(width, height);
        fill_img(&images[i], model_par(parameter[i]), NULL);
    }
    struct mim_model *model = mim_model_create(
            size, parameter, images);
    
    for(size_t i = 0; i < size; i++) {
        images[i]->destroy(&images[i]);
    }
    return model;
}
