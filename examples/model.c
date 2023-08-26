/* Public API */
#include "mim.h"
/* C standard library */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double get_count(const double density) {
	return 100. - 5. * density;
}

int main(void) {
	time_t t;
	/* Intializes random number generator */
	srand((unsigned) time(&t));
	
	const size_t width = 6;
	const size_t height = 4;
	const size_t size = 7;
	
	/* Model parameters */
	double parameter[size];
	struct mim_img *images[size];
	
	const double pmin = 1.0;
	const double pmax = 2.8;
	const double pdl = (pmax - pmin) / (size - 1);
	
	/* Model of images */	
	size_t i, j, k;
	for(k = 0; k < size; k++) {
		parameter[k] = 1.0 + k * pdl;
		images[k] = mim_img_empty(width, height);
	}
	/* Observation */
	struct mim_img *observation = mim_img_empty(width, height);
	
	/* Fill with abstract data */
	for(i = 0; i < width; i++) {
		for(j = 0; j < height; j++) {			
			double value;
			for(k = 0; k < size; k++) {
				value = get_count(parameter[k]);
				images[k]->set(images[k], i, j, value);
			}
			
			const double rdensity1 = parameter[rand() % size];
			const double rdensity2 = parameter[rand() % size];
			
			value = get_count(0.5 * (rdensity1 + rdensity2));
			observation->set(observation, i, j, value);
		}
	}	
	struct mim_model *model = mim_model_create(
			size, parameter, images);
	
	/* Inverted result */
	struct mim_img *result = mim_img_empty(width, height);	
	
	
	enum mim_return rc = mim_model_invert(
			result, model, observation, NULL);
	
	if(rc == MIM_FAILURE) {
		return 1;
	}
	
	for(i = 0; i < width; i++) {
		for(j = 0; j < height; j++) {			
			printf("(%ld, %ld)", i, j);			
			for(k = 0; k < size; k++) {
				printf(" %g[%g]", images[k]->get(images[k], i, j), parameter[k] );
			}
			
			printf(" -> %g[%g]\n", observation->get(observation, i, j), result->get(result, i, j));
		}
	}
	
	/* Free allocated memory */
	for(k = 0; k < size; k++) {
		images[k]->destroy(&images[k]);
	}
	model->destroy(&model);
	result->destroy(&result);
	observation->destroy(&observation);
	
	return 0;
}
