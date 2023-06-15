#ifndef mim_h
#define mim_h

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>


/* Version macros */
#define MIM_VERSION_MAJOR 0
#define MIM_VERSION_MINOR 1
#define MIM_VERSION_PATCH 0

enum mim_return {
	MIM_SUCCESS = 0,
	MIM_FAILURE
};


/* ============================================================================
 * Image interface
 * ============================================================================
 */
struct mim_img;

typedef double mim_img_get_t(const struct mim_img *self,
		const size_t i, const size_t j);

typedef void mim_img_set_t(struct mim_img *self,
		const size_t i, const size_t j, const double value);

typedef double *mim_img_ref_t(struct mim_img *self,
		const size_t i, const size_t j);

typedef void mim_img_destroy(struct mim_img **self_ptr);

struct mim_img {
	/* Meta data */
	const size_t width;
	const size_t height;
	
	/* Get/setters */
	mim_img_get_t *get;
	mim_img_set_t *set;
	mim_img_ref_t *ref;
	
	/* Free memory */
	mim_img_destroy *destroy;
};

/* Create an empy image */
struct mim_img *mim_img_empty(const size_t width,
		const size_t height);

/* Create a zeroed image */
struct mim_img *mim_img_zeros(const size_t width,
		const size_t height);


#ifdef __cplusplus
}
#endif

#endif