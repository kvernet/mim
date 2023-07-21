/* Public API */
#include "mim.h"
/* C standard library */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct mim_img *get_random_image(struct mim_img *image,
        struct mim_prng *prng);

double model_parametric(const double par) {
    return 9 - par * par;
}

int main(int argc, char **argv) {
    if(argc < 2) {
        fputs("Usage:\n", stderr);
        fputs("\t", stderr);
        fputs(argv[0], stderr);
        fputs(" N\n", stderr);
        return EXIT_FAILURE;
    }
    const long N = (long)strtod(argv[1], NULL);
    
    /* Image shape */
    const size_t width = 201, height = 101;
    
    /* Model */
    const double pmin = 1.0, pmax = 3.0;
    const int np = 21;
    const double pdelta = (pmax - pmin) / (np - 1);
    double parameter[np];
    struct mim_img *images[np];    
    for(int ip = 0; ip < np; ip++) {
        parameter[ip] = pmin + ip * pdelta;
        images[ip] = mim_img_empty(width, height);
        for(size_t iw = 0; iw < width; iw++) {
            for(size_t ih = 0; ih < height; ih++) {
                images[ip]->set(images[ip], iw, ih, 
                    model_parametric(parameter[ip]));
            }
        }
    }
    struct mim_model *model = 
            mim_model_create(np, parameter, images);
    
    /* Observation */
    const double tparameter = 1.8;
    struct mim_img *obs = NULL;
    obs = mim_img_empty(width, height);
    for(size_t iw = 0; iw < width; iw++) {
        for(size_t ih = 0; ih < height; ih++) {
            obs->set(obs, iw, ih, 
                    model_parametric(tparameter));
        }
    }
    
    /* Pseudo random number generator */
    struct mim_prng *prng = mim_prng_init(0);
    
    double psum = 0., psum2 = 0.;
    double vsum = 0., vsum2 = 0.;
    
    for(long i = 0; i < N; i++) {
        struct mim_img *image = mim_img_empty(width, height);
        // get random observation
        struct mim_img *robs = 
                get_random_image(obs, prng);
        
        // invert random observation
        enum mim_return mrc = mim_model_invert(
                image, model, robs);
        
        if(mrc == MIM_SUCCESS) {
            const double par = image->get(image, 0, 0);
            const double rvalue = robs->get(robs, 0, 0);
            
            psum += par;
            psum2 += par * par;
            vsum += rvalue;
            vsum2 += rvalue * rvalue;
            
            printf("par = %3.3f [rvalue = %3.1f]\n", 
                 par, rvalue);
        }
        else {
            fputs("error[min] - could not invert image\n", stderr);
            return EXIT_FAILURE;
        }
        
        robs->destroy(&robs);
        image->destroy(&image);
    }
    
    psum /= N;
    vsum /= N;
    printf("*** par = %3.3f : par_hat = %3.3f +- %3.3f [value = %3.3f +- %3.3f] ***\n", 
                 tparameter,
                 psum, sqrt((psum2 / N - psum*psum) / N), 
                 vsum, sqrt((vsum2 / N - vsum*vsum) / N));
    
    
        
    /* Free allocated memory */
    for(int i = 0; i < np; i++) {
        images[i]->destroy(&images[i]);
    }
    model->destroy(&model);
    obs->destroy(&obs);
    prng->destroy(&prng);
}

struct mim_img *get_random_image(struct mim_img *image,
        struct mim_prng *prng) {
    struct mim_img *result = 
            mim_img_empty(image->width, image->height);
    
    for(size_t iw = 0; iw < image->width; iw++) {
        for(size_t ih = 0; ih < image->height; ih++) {
            const double value = image->get(image, iw, ih);
            const double rvalue = prng->poisson(prng, value);
            result->set(result, iw, ih, rvalue);
        }
    }
    
    return result;
}
