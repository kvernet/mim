/* Public API. */
#include "mim.h"
/* C stantard library */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Image implementation.
 * ============================================================================
 */
struct img {
    /* Public interface */
    struct mim_img pub;
    
    /* Placeholder for data. */
    double data[];
};

static double img_get(const struct mim_img *self,
        const size_t i, const size_t j) {
    struct img *img = (void *)self;
    return img->data[i * self->height + j];
}

static void img_set(struct mim_img *self,
        const size_t i, const size_t j,
        const double value) {
    struct img *img = (void *)self;
    img->data[i * self->height + j] = value;
}

static double *img_ref(struct mim_img *self,
        const size_t i, const size_t j) {
    struct img *img = (void *)self;
    return img->data + i * self->height + j;
}

static void img_destroy(struct mim_img **self_ptr) {
    if(self_ptr == NULL || *self_ptr == NULL) return;
    free(*self_ptr);
    self_ptr = NULL;
}

static struct mim_img *img_create(const size_t width,
        const size_t height, void *(*allocator)(size_t)) {
    /* Allocate memory */
    struct img *img = allocator( sizeof(*img) + width * height * sizeof(*img->data) );
    if(img == NULL) return NULL;
    
    /* Set metadata */
    *(size_t*)&img->pub.width = width;
    *(size_t*)&img->pub.height = height;
    
    /* Set functions */
    img->pub.get = &img_get;
    img->pub.set = &img_set;
    img->pub.ref = &img_ref;
    img->pub.destroy = &img_destroy;
    
    return &img->pub;
}

struct mim_img *mim_img_empty(const size_t width,
        const size_t height) {
    return img_create(width, height, &malloc);
}

static void *img_alloc(size_t size) { return calloc(1, size); }

struct mim_img *mim_img_zeros(const size_t width,
        const size_t height) {
    return img_create(width, height, &img_alloc);
}

/* ============================================================================
 * Model implementation.
 * ============================================================================
 */

/* Number of splines coefficients. */
#define N_COEFS 3


struct model {
    /* Public interface */
    struct mim_model pub;
    
    /* Number of model samples per image point. */
    size_t depth;
    
    /* Pointer to the sampled parameter values.
     *
     * Note: parameter values are structured as a vector of size *depth*.
     */
    double *parameter;
    
    /* Pointer to the monotone cubic splines coefficients
     *
     * Note: coefficients are structured as an array of dimensions (width,
     * height, depth, N_COEFS), where the two lower indices store splines
     * coefficients.
     */
    double *spline;
    
    /* Placeholder for data (see above).*/
    double data[];
};

static void model_destroy(struct mim_model **self_ptr) {
    if(self_ptr == NULL || *self_ptr == NULL) return;
    free(*self_ptr);
    self_ptr = NULL;
}

static void spline_initialise(size_t n,
        const double *x, const double *y, double *m);
    
/* Create a model object from a parametric collection of images. */
struct mim_model *mim_model_create(const size_t depth,
        const double parameter[],
        struct mim_img *images[]) {
    if(depth <= 0) return NULL;
    /* Allocate memory */
    const size_t width = images[0]->width;
    const size_t height = images[0]->height;
    /* Check the shape of all the images */
    size_t i;
    for(i = 1; i < depth; i++) {
        if(images[i]->width != width || images[i]->height != height) {
            return NULL;
        }
    }
    struct model *model;
    const size_t memsize = sizeof(*model) +
            depth * sizeof(*model->parameter) +
            width * height * depth * N_COEFS * sizeof(*model->spline);
    model = malloc(memsize);
    if(model == NULL) return NULL;
    
    /* Set metadata */
    *(size_t*)&model->pub.width = width;
    *(size_t*)&model->pub.height = height;
    
    double pmin = parameter[0], pmax = parameter[depth - 1];
    if(pmax < pmin) {
        double tmp = pmin;
        pmin = pmax;
        pmax = tmp;
    }
    *(double*)&model->pub.pmin = pmin;
    *(double*)&model->pub.pmax = pmax;
    
    model->pub.destroy = &model_destroy;
    
    /* Map data pointers */
    model->depth = depth;
    model->parameter = model->data;
    model->spline = model->parameter + depth;
    
    /* Initialize spline coefficients */
    memcpy(model->parameter, parameter, depth * sizeof(*parameter));
    
    double *spline = model->spline;
    for(i = 0; i < width; i++) {
        size_t j;
        for(j = 0; j < height; j++, spline += N_COEFS * depth) {
            double *y = spline;
            size_t k;
            for(k = 0; k < depth; k++) {
                const struct mim_img *img = images[k];
                y[k] = img->get(img, i, j);
            }
            /* Forward interpolation */
            double *mf = y + depth;
            spline_initialise(depth, parameter, y, mf);
            
            /* Backward interpolation */
            double *mb = mf + depth;
            spline_initialise(depth, y, parameter, mb);
        }
    }
    
    return &model->pub;
}

/* Compute the derivative coefficients for the monotone cubic splines
 * interpolation
 *
 * The derivative cooefficients are computed using the method of Fritsch and
 * Butland. For boundary conditions a 3 points finite difference is used.
 *
 * References:
 *  F. N. Fristch and J. Butland, SIAM J. Sci. Stat. Comput. (1984)
 */
static double diff3(
        double x0, double x1, double x2, double y0, double y1, double y2) {
    /* Evaluate the derivative at x0 using 3 neighbouring values. */
    const double h1 = x1 - x0;
    const double h2 = x2 - x0;
    const double delta = h1 * h2 * (h2 - h1);
    const double c1 = h2 * h2 / delta;
    const double c2 = -h1 * h1 / delta;
    const double c0 = -(c1 + c2);
    return c0 * y0 + c1 * y1 + c2 * y2;
}

static void spline_initialise(size_t n,
        const double *x, const double *y,
        double *m) {
    if(n == 1) {
        m[0] = 0;
    } else if(n == 2) {
        const double d = (y[1] - y[0]) / (x[1] - x[0]);
        m[0] = m[1] = d;
    } else {
        size_t i;
        for(i = 1; i < n - 1; i++) {
            const double h1 = x[i] - x[i - 1];
            if(h1 == 0.0) {
                m[i] = 0.0;
                continue;
            }
            const double h2 = x[i + 1] - x[i];
            if(h2 == 0.0) {
                m[i] = 0.0;
                continue;
            }
            const double S1 = (y[i] - y[i - 1]) / h1;
            const double S2 = (y[i + 1] - y[i]) / h2;
            
            const double tmp = S1 * S2;
            if(tmp > 0) {
                const double a =
                        (h1 + 2 * h2) / (3 * (h1 + h2));
                m[i] = tmp / ((1 - a) * S1 + a * S2);
            } else {
                m[i] = 0.;
            }
        }
        
        m[0] = diff3(x[0], x[1], x[2], y[0], y[1], y[2]);
        
        m[n - 1] = diff3(
                x[n - 1], x[n - 2], x[n - 3],
                y[n - 1], y[n - 2], y[n - 3]);
    }
}

/* Interpolation using monotone cubic splines (see above). */
static double spline_interpolate(size_t n,
        const double * x, const double * y,
        const double * m, double xi) {
    if(n == 1) {
        return y[0];
    }
    
    /* Let us first check if xi is within outer bounds. */
    const double sgn = (x[1] > x[0]) ? 1.0 : -1.0;
    if(sgn * (xi - x[0]) <= 0.0) {
        return y[0];
    } else if(sgn * (xi - x[n - 1]) >= 0.0) {
        return y[n - 1];
    }
    
    /* Binary search of bounding interval.
     *
     * Note that the checks above and below ensure that the
     * delivered i0 value is in [0, n - 2].
     */
    size_t i0 = 0, i1 = n - 1;
    while(i1 - i0 > 1) {
        size_t i2 = (i0 + i1) / 2;
        const double x2 = x[i2];
        if(xi == x2) {
            i0 = i2;
            break;
        } else if(sgn * (xi - x2) > 0.0) {
            i0 = i2;
        } else {
            i1 = i2;
        }
    }
    
    /* Hermite polynomials interpolation using the 1st derivative.
     *
     * Reference:
     *   https://fr.wikipedia.org/wiki/Spline_cubique_d%27Hermite
     */
    const double dx = x[i0 + 1] - x[i0];
    const double t = (xi - x[i0]) / dx;
    const double p0 = y[i0];
    const double p1 = y[i0 + 1];
    const double m0 = m[i0] * dx;
    const double m1 = m[i0 + 1] * dx;
    const double c2 = -3 * (p0 - p1) - 2 * m0 - m1;
    const double c3 = 2 * (p0 - p1) + m0 + m1;
    
    return p0 + t * (m0 + t * (c2 + t * c3));
}

/* Get a snapshot of the model for a given parameter value (by interpolation).
 */
enum mim_return mim_model_get(
        struct mim_img *image,
        const struct mim_model *pub,
        double parameter) {
    
    if((image->height != pub->height) ||
            (image->width != pub->width)) {
                return MIM_FAILURE;
    }
    
    struct model *model = (void *)pub;
    const size_t width = pub->width;
    const size_t height = pub->height;
    const size_t depth = model->depth;
    double * spline = model->spline;
    size_t i;
    for(i = 0; i < width; i++) {
        size_t j;
        for(j = 0; j < height; j++, spline += N_COEFS * depth) {
            const double *count = spline;
            const double *mf = count + depth;
            const double value = spline_interpolate(
                    depth, model->parameter,
                    count, mf, parameter);
            
            image->set(image, i, j, value);
        }
    }
    
    return MIM_SUCCESS;
}

/* Get parameter values for a given model and observation (i.e. invert the
 * observation for the model).
 */
enum mim_return mim_model_invert(
        struct mim_img *image,
        const struct mim_model *pub,
        const struct mim_img *observation,
        const struct mim_img *filter) {
    
    struct model *model = (void *)pub;
    const size_t width = pub->width;
    const size_t height = pub->height;
    
    if(image->width != width || image->height != height ||
            observation->width != width || observation->height != height) {
        return MIM_FAILURE;
    }    
    if(filter != NULL) {
        if(filter->width != width || filter->height != height)
            return MIM_FAILURE;
    }
    
    const size_t depth = model->depth;
    double *spline = model->spline;
    size_t i;
    for(i = 0; i < width; i++) {
        size_t j;
        for(j = 0; j < height; j++, spline += N_COEFS * depth) {
            double fvalue = 1.0;
            if(filter != NULL) {
                fvalue = filter->get(filter, i, j);
            }
            if(fvalue > 0) {
                const double *count = spline;
                const double *mb = count + 2 * depth;
                const double value = spline_interpolate(
                        depth, count, model->parameter,
                        mb, observation->get(observation, i, j));
                
                image->set(image, i, j, value);
            }
            else {
                image->set(image, i, j, BIN_FILTER_VALUE);
            }
        }
    }
    
    return MIM_SUCCESS;
}

/* Integrate bins to get the minimum value */
static enum mim_return mim_integrate_bins_for_min(
        size_t *bins, double *value,
        const size_t i, const size_t j,
        const struct mim_img *observation,
        const struct mim_img *filter,
        const double min_value);

/* Integrate bins */
static double integrate_bins(
        const size_t i, const size_t j,
        const size_t bins,
        const struct mim_img *image,
        const struct mim_img *filter);

static enum mim_return mim_compute_stats(
        struct mim_img *image,
        const struct mim_img *bin_image,
        const struct mim_img *filter,
        const double sigma);

/* Get parameter values for a given model and observation with
 * a minimum value
 */
enum mim_return mim_model_min_invert(
        struct mim_img *image,
        struct mim_img *bin_image,
        struct mim_img *value_image,
        const struct mim_model *model,
        const struct mim_img *observation,
        const struct mim_img *filter,
        const double min_value,
        const double sigma) {
    
    if(model == NULL || observation == NULL) return MIM_FAILURE;
    const size_t width = observation->width;
    const size_t height = observation->height;
    
    if(image->width != width || image->height != height ||
            bin_image->width != width || bin_image->height != height ||
            value_image->width != width || value_image->height != height ||
            observation->width != width || observation->height != height) {
        return MIM_FAILURE;
    }    
    if(filter != NULL) {
        if(filter->width != width || filter->height != height)
            return MIM_FAILURE;
    }
    
    
    const double *parameter = ((struct model*)model)->parameter;
    const size_t size = ((struct model*)model)->depth;
    /* Get the the model images */
    struct mim_img *model_images[size];
    
    enum mim_return mrc;
    
    /* 1D images to invert a bin */
    struct mim_img *tmp_image = mim_img_empty(1, 1);
    struct mim_img *tmp_observation = mim_img_empty(1, 1);
    struct mim_img *tmp_images[size];
    for(size_t i = 0; i < size; i++) {
        tmp_images[i] = mim_img_empty(1, 1);
        model_images[i] = mim_img_empty(width, height);
        mrc = mim_model_get(
                model_images[i], model, parameter[i]);
        if(mrc == MIM_FAILURE) return mrc;
    }
    
    
    size_t i, j;
    for(i = 0; i < width; i++) {
        for(j = 0; j < height; j++) {
            double fvalue = 1.0;
            if(filter != NULL) {
                fvalue = filter->get(filter, i, j);
            }
            
            if(fvalue > 0) {
                size_t bins;
                double value, par;
                /* Integrate nearest bins to get the minimum value */                
                mim_integrate_bins_for_min(&bins, &value,
                        i, j, observation, filter, min_value);
                
                if(bin_image != NULL) {
                    bin_image->set(bin_image, i, j, bins);
                }
                if(value_image != NULL) {
                    value_image->set(value_image, i, j, value);
                }
                
                /* Invert the model for that bin */
                if(image != NULL) {
                    tmp_observation->set(tmp_observation, 0, 0, value);
                    
                    for(size_t ip = 0; ip < size; ip++) {
                        const double v = integrate_bins(
                                i, j, bins, model_images[ip], filter);
                        tmp_images[ip]->set(tmp_images[ip], 0, 0, v);
                    }
                    
                    struct mim_model *tmp_model = mim_model_create(
                            size, parameter, tmp_images);
                    
                    mrc = mim_model_invert(
                            tmp_image, tmp_model, tmp_observation, NULL);
                    if(mrc == MIM_FAILURE) {
                        tmp_model->destroy(&tmp_model);
                        return mrc;
                    }
                    
                    par = tmp_image->get(tmp_image, 0, 0);
                    image->set(image, i, j, par);
                    tmp_model->destroy(&tmp_model);
                }
            }
            else {
                if(bin_image != NULL) {
                    bin_image->set(bin_image, i, j, 0);
                }
                if(value_image != NULL) {
                    value_image->set(value_image, i, j, BIN_FILTER_VALUE);
                }
                if(image != NULL) {
                    image->set(image, i, j, BIN_FILTER_VALUE);
                }
            }
        }
    }
    
    /* Compute statistics */
    if(image != NULL) {
        mrc = mim_compute_stats(
                image, bin_image, filter, sigma);
        if(mrc == MIM_FAILURE) return MIM_FAILURE;
    }
    
    tmp_image->destroy(&tmp_image);
    tmp_observation->destroy(&tmp_observation);
    for(size_t i = 0; i < size; i++) {
        tmp_images[i]->destroy(&tmp_images[i]);
        model_images[i]->destroy(&model_images[i]);
    }    
    
    return MIM_SUCCESS;
}

enum mim_return mim_integrate_bins_for_min(
        size_t *bins, double *value,
        const size_t i, const size_t j,
        const struct mim_img *image,
        const struct mim_img *filter,
        const double min_value) {
    
    double sum = image->get(image, i, j);
    
    long n = 1;
    while(sum < min_value) {
        for(long k = -n; k < n + 1; k++) {
            long step = 0;
            if(abs(k) == n) step = 1;
            else step = 2*n;
                        
            for(long l = -n; l < n + 1; l+=step) {
                if(i + k >= 0 && i + k < image->width &&
                        j + l >= 0 && j + l < image->height) {
                    double fvalue = 1.0;
                    if(filter != NULL) {
                        fvalue = filter->get(filter, i + k, j + l);
                    }
                    if(fvalue > 0) {
                        sum += image->get(image, i + k, j + l);
                    }
                }
            }
        }
        n++;
    }
	
	*bins  = 2 * n - 1;
	*value = sum;    
    
    return MIM_SUCCESS;
}

double integrate_bins(
        const size_t i, const size_t j,
        const size_t bins,
        const struct mim_img *image,
        const struct mim_img *filter) {
    double sum = 0.;
    long n = (bins - 1) / 2;
    for(long k = -n; k < n + 1; k++) {
        for(long l = -n; l < n + 1; l++) {
            if(i + k >= 0 && i + k < image->width &&
                    j + l >= 0 && j + l < image->height) {
                double fvalue = 1.0;
                if(filter != NULL) {
                    fvalue = filter->get(filter, i + k, j + l);
                }
                if(fvalue > 0) {
                    sum += image->get(image, i + k, j + l);
                }
            }
        }
    }
    return sum;
}

enum mim_return mim_compute_stats(
        struct mim_img *image,
        const struct mim_img *bin_image,
        const struct mim_img *filter,
        const double sigma) {
    const size_t width = image->width;
    const size_t height = image->height;
    
    if(bin_image->width != width || bin_image->height != height) {
        return MIM_FAILURE;
    }
    
    struct mim_img *sum_image = mim_img_zeros(width, height);
    struct mim_img *weight_image = mim_img_zeros(width, height);
    
    double fvalue;
    size_t i, j;
    for(i = 0; i < width; i++) {
        for(j = 0; j < height; j++) {
            fvalue = 1.0;
            if(filter != NULL) {
                fvalue = filter->get(filter, i, j);
            }
            if(fvalue < 0) continue;            
            
            const size_t i0 = i;
            const size_t j0 = j;
            const double par = image->get(image, i, j);
            const size_t bins = (size_t)bin_image->get(bin_image, i, j);
            long n = (bins - 1) / 2;
            for(long k = -n; k < n + 1; k++) {
                for(long l = -n; l < n + 1; l++) {
                    if(i + k >= 0 && i + k < width &&
                            j + l >= 0 && j + l < height) {
                        double fvalue = 1.0;
                        if(filter != NULL) {
                            fvalue = filter->get(filter, i + k, j + l);
                        }
                        if(fvalue > 0) {
                            const double t1 = (i + k - i0);
                            const double t2 = (j + l - j0);
                            const double d2 = t1*t1 + t2*t2;
                            const double arg = d2 / sigma;
                            const double w = exp(-0.5 * arg);
                            
                            double s = sum_image->get(sum_image, i + k, j + l);
                            sum_image->set(sum_image, i + k, j + l, s + w*par);
                            
                            s = weight_image->get(weight_image, i + k, j + l);
                            weight_image->set(weight_image, i + k, j + l, s + w);
                        }
                    }
                }
            }
        }
    }
    
    for(i = 0; i < width; i++) {
        for(j = 0; j < height; j++) {
            double fvalue = 1.0;
            if(filter != NULL) {
                fvalue = filter->get(filter, i, j);
            }
            if(fvalue > 0) {
                const double w = weight_image->get(weight_image, i, j);
                const double mean = sum_image->get(sum_image, i, j) / w;
                image->set(image, i, j, mean);
            }
            else {
                image->set(image, i, j, BIN_FILTER_VALUE);
            }
        }
    }
    
    /* Free allocated memory */
    sum_image->destroy(&sum_image);
    weight_image->destroy(&weight_image);
    
    return MIM_SUCCESS;
}

/* ============================================================================
 * Pseudo random number generator interface
 * ============================================================================
 */
struct prng {
    struct mim_prng pub;
    
    const double weight;    /* Monte Carlo weight */
};

static double uniform1(struct mim_prng *prng);
static double normal(struct mim_prng *prng,
        const double mu, const double sigma);
static double poisson(struct mim_prng *prng,
        const double lambda);
static double weight(struct mim_prng *prng);

static void set_seed(struct mim_prng *prng, const unsigned long s) {
    *(unsigned long*)&prng->seed = s;
    /* Set seed */
    srand(prng->seed);
}

static void mim_prng_destroy(struct mim_prng **self_ptr) {
    if(self_ptr == NULL || *self_ptr == NULL) return;
    free(*self_ptr);
    self_ptr = NULL;
}

struct mim_prng *mim_prng_init(const unsigned long seed) {
    struct prng *prng = malloc(sizeof(*prng));
    
    if(seed > 0) {
        *(unsigned long*)&prng->pub.seed = seed;
    }
    else {
        FILE *stream = fopen("/dev/urandom", "rb");
        if(stream == NULL) *(unsigned long*)&prng->pub.seed = time(NULL);
        else {
            unsigned int seeds[2];
            fread(seeds, sizeof seeds, 1, stream);
            *(unsigned long*)&prng->pub.seed = seeds[0];
            fclose(stream);
        }
    }
    
    prng->pub.set_seed = &set_seed;
    prng->pub.destroy = &mim_prng_destroy;
    
    prng->pub.uniform1 = &uniform1;
    prng->pub.normal = &normal;
    prng->pub.poisson = &poisson;
    prng->pub.weight = &weight;    
    
    /* Default weight */
    *(double*)&prng->weight = 1.;
    
    /* Set seed */
    srand(prng->pub.seed);
    
    return &prng->pub;
}



static double uniform1(struct mim_prng *prng) {
    struct prng *p = (void*)prng;
    *(double*)&p->weight = 1;
    return (double)rand() / (double)RAND_MAX;
}

double normal(struct mim_prng *prng,
        const double mu, const double sigma) {
    struct prng *p = (void*)prng;
    
    if(sigma == 0) {
        *(double*)&p->weight = 1.;
        return mu;
	}
	
	const double u = uniform1(prng);
	const double v = uniform1(prng);
	
	const double two_pi = 2 * M_PI;
	const double mag = sqrt(-2 * log(u));
	
	double x = sigma * mag * cos(two_pi * v) + mu;
	
	// compute weight = inv(pdf)
	const double xi = (x - mu) / sigma;
	const double nume = exp(-0.5 * xi * xi);
	const double deno = sigma * sqrt(two_pi);
	*(double*)&p->weight = deno / nume;
	
	return x;
}

static double poisson_knuth(struct mim_prng *prng,
        const double lambda);
static double poisson_cook(struct mim_prng *prng,
        const double lambda);
double poisson(struct mim_prng *prng,
        const double lambda) {
    if(abs(lambda) >= DBL_MAX) return DBL_MAX;
    else if(lambda < 30) return poisson_knuth(prng, lambda);
    else return poisson_cook(prng, lambda);
}

static double weight(struct mim_prng *prng) {
    struct prng *p = (void*)prng;
    return p->weight;
}

long long compute_factorial_minus_one(long long k) {
    long long fact = 1;
    for(long long i = 2; i < k - 1; i++) {
        fact *= i;
    }
    return fact;
}

double compute_poisson_weight(const double lambda,
        const long long k, const double x) {
    long long fact = compute_factorial_minus_one(k);
    
    const double pdf = exp(-lambda) * pow(lambda, x) / (double)fact;
    return 1 / pdf;
}

// algorithm https://fr.wikipedia.org/wiki/Loi_de_Poisson
double poisson_knuth(struct mim_prng *prng,
        const double lambda) {
    struct prng *rng = (void*)prng;
    
    double p = 1;
    long long k = 0;
    
    while(p > exp(-lambda)) {
        const double u = uniform1(prng);
        if((u <= 0.) || (u >= 1.)) continue;
        
        p *= u;
        k += 1;
    }
    
    double x = (double) (k - 1);
    
    *(double*)&rng->weight = compute_poisson_weight(
            lambda, k, x);
    
    return x;
}
// reference: https://www.johndcook.com/blog/2010/06/14/generating-poisson-random-values/
double poisson_cook(struct mim_prng *prng,
        const double lambda) {
    struct prng *rng = (void*)prng;
    
    const double c = 0.767 - 3.36 / lambda;
    const double beta = M_PI / sqrt(3.0 * lambda);
    const double alpha = beta * lambda;
    const double k = log(c) - lambda - log(beta);
    
    while(true) {
        const double u = uniform1(prng);
        const double x = (alpha - log((1.0 - u)/u)) / beta;
        const long n = floor(x + 0.5);
        if(n < 0) continue;
        
        const double v = uniform1(prng);
        const double y = alpha - beta * x;
        const double t = 1.0 + exp(y);
        const double lhs = y + log(v / (t*t));
        // lgamma c++ function https://en.cppreference.com/w/cpp/numeric/math/lgamma
        // N.B log(n!) ~= lgamma(n + 1)
        const double rhs = k + n*log(lambda) - lgamma(n + 1);
        
        if(lhs <= rhs) {
            /* FIXME */
            *(double*)&rng->weight = compute_poisson_weight(
                    lambda, n + 1, n);
            
            return n;
        }
    }
}
