//
// Created by vmachado on 2/11/20.
//

#ifndef fc_layer_h
#define fc_layer_h

#include "matrix.h"

typedef struct _fc_layer {
    matrix *weights, *input;
} fc_layer;

fc_layer* fc_alloc(int in_dim, int out_dim);
void fc_free(fc_layer *layer);

matrix* fc_forward(fc_layer *layer, matrix *raw_input);
matrix* fc_backward(fc_layer *layer, matrix *dout, float lambda_reg, float l_rate);

#endif //fc_layer_h
