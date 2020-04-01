//
// Created by vmachado on 2/11/20.
//

#ifndef fc_layer_h
#define fc_layer_h

#include "matrix.h"
#include <stdbool.h>

typedef struct _fc_layer {
    bool relu;
    matrix *weights, *gamma, *beta;
    matrix *run_mean, *run_var, *stddev_inv;
    matrix *input, *input_norm;
} fc_layer;

fc_layer* fc_alloc(int in_dim, int out_dim, bool relu);
void fc_free(fc_layer *layer);
void fc_update_status(fc_layer *layer, matrix *mean, matrix *variance);

matrix* fc_forward(fc_layer *layer, matrix *raw_input, bool training);
matrix* fc_backward(fc_layer *layer, matrix *dout, float lambda_reg, float l_rate);

#endif //fc_layer_h
