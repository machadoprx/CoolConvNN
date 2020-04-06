#ifndef conv_layer_h
#define conv_layer_h

#include "matrix.h"
#include "image.h"
#include <stdbool.h>

typedef struct _conv_layer {
    int in_c, out_c;
    int in_w, in_h;
    int stride, f_size, padd;
    int col_w, col_h, col_c;
    int in_dim, out_dim, col_dim;
    int *activations;
    matrix *filters, *gamma, *beta;
    matrix *run_mean, *run_var, *stddev_inv;
    matrix *input_col, *out_norm;
} conv_layer;

conv_layer* conv_alloc(int in_c, int in_w, int in_h, int out_c, int f_size, int stride, int padd);
void conv_free(conv_layer *layer);
matrix* conv_forward(conv_layer *layer, matrix *raw_input, bool training);
matrix* conv_backward(conv_layer *layer, matrix *dout, float l_rate);

#endif