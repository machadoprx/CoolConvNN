#ifndef pool_layer_h
#define pool_layer_h

#include "matrix.h"

typedef struct _pool_layer {
    int chan, in_w, in_h, in_dim, padd, f_size, stride;
    int out_w, out_h, out_dim;
    int *indexes;
} pool_layer;

pool_layer* pool_alloc(int chan, int in_w, int in_h, int f_size, int stride, int padd);
void pool_free(pool_layer *layer);

matrix* pool_forward(pool_layer *layer, matrix *raw_input);
matrix* pool_backward(pool_layer *layer, matrix *dout);

#endif