/* Public API. */
#include "mim.h"
/* C stantard library */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>

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
        const struct mim_img *observation) {
    
    if((observation->height != pub->height) ||
            (observation->width != pub->width) ||
            (image->height != pub->height) ||
            (image->width != pub->width)) {
        return MIM_FAILURE;
    }
    
    struct model *model = (void *)pub;
    const size_t width = pub->width;
    const size_t height = pub->height;
    const size_t depth = model->depth;
    double *spline = model->spline;
    size_t i;
    for(i = 0; i < width; i++) {
        size_t j;
        for(j = 0; j < height; j++, spline += N_COEFS * depth) {
            const double *count = spline;
            const double *mb = count + 2 * depth;
            const double value = spline_interpolate(
                    depth, count, model->parameter,
                    mb, observation->get(observation, i, j));
            
            image->set(image, i, j, value);
        }
    }
    
    return MIM_SUCCESS;
}
