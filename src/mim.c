/* Public API. */
#include "mim.h"
/* C stantard library */
#include <stdio.h>
#include <stdlib.h>

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
		const size_t height,
		void *(*allocator)(size_t)) {
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
	return img_create(
		width,
		height,
		&malloc
	);
}

static void *img_alloc(size_t size) { return calloc(1, size); }

struct mim_img *mim_img_zeros(const size_t width,
		const size_t height) {
	return img_create(
		width,
		height,
		&img_alloc
	);
}
