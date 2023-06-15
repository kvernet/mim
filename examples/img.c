/* Public API */
#include "mim.h"
/* C standard library */
#include <stdio.h>

int main(void) {
	/* Image dimension */
	const size_t width = 3;
	const size_t height = 2;
	
	/* Create a zeroed image */
	struct mim_img *zeros = mim_img_zeros(width, height);
	
	/* Access each pixel of the zeroed image */
	size_t i, j;
	for(i = 0; i < width; i++) {
		for(j = 0; j < height; j++) {
			const double zv = zeros->get(zeros, i, j);
			
			printf("zeros(%ld, %ld) = %g\n", i, j, zv);
		}
	}
	
	/* Free allocated memory */
	zeros->destroy(&zeros);
	
	return 0;
}
