#ifndef png2array_h
#define png2array_h
#include "lodepng.h"
#include "../utils.h"
#include <stdio.h>

void load_data(const char* path, float*** data, float** mean, float** var, int *labels, int *label_samples, int *in_dim);
int *gen_targets(int num_labels, int label_samples);
float *decode_png(const char* path, int *w, int *h);

#endif