#ifndef bn_layer_h
#define bn_layer_h

#include "matrix.h"
#include "utils.h"
#include <stdbool.h>

typedef struct _bn_layer {
    int in_channels, in_spatial;
    matrix *gamma, *beta, *variance;
    matrix *run_var, *run_mean;
    matrix *out_cache;
} bn_layer;

bn_layer *bn_alloc(int in_channels, int in_spatial);
void bn_free(bn_layer *layer);

matrix* bn_forward(bn_layer *layer, matrix *input, bool training);
matrix* bn_backward(bn_layer *layer, matrix *dout, float l_rate);

#endif