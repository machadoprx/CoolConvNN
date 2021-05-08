//
// Created by vmachado on 2/11/20.
//

#include "fc_layer.h"

fc_layer* fc_alloc(int in_dim, int out_dim) {

    fc_layer* layer = aalloc(sizeof(*layer));
    layer->weights = matrix_alloc(in_dim, out_dim);

    layer->input = NULL;

    randomize(layer->weights, 0.0f, sqrtf(2.0f / (float)in_dim));

    return layer;
}

void fc_free(fc_layer *layer) {
    matrix_free(layer->weights);
    matrix_free(layer->input);
    free(layer);
}

matrix* fc_forward(fc_layer *layer, matrix *raw_input) {

    matrix_free(layer->input);

    layer->input = mat_copy(raw_input);

    return multiply(raw_input, layer->weights, false, false, 
                    raw_input->rows, layer->weights->columns, raw_input->columns);
}

matrix* fc_backward(fc_layer *layer, matrix *dout, float lambda_reg, float l_rate) {

    matrix *dweights = multiply(layer->input, dout, true, false, layer->input->columns, dout->columns, layer->input->rows);
    apply_sum(dweights, layer->weights, lambda_reg);

    matrix *dinput = multiply(dout, layer->weights, false, true, dout->rows, layer->weights->rows, dout->columns);

    apply_sum(layer->weights, dweights, -l_rate);

    matrix_free(dweights);

    return dinput;
}
