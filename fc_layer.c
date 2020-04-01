//
// Created by vmachado on 2/11/20.
//

#include "fc_layer.h"

fc_layer* fc_alloc(int in_dim, int out_dim, bool relu) {

    fc_layer* layer = aligned_alloc(CACHE_LINE, sizeof(*layer));
    layer->weights = matrix_alloc(in_dim, out_dim);
    layer->gamma = matrix_alloc(1, in_dim);
    layer->beta = matrix_alloc(1, in_dim);
    layer->run_mean = matrix_alloc(1, in_dim);
    layer->run_var = matrix_alloc(1, in_dim);

    layer->relu = relu;
    layer->input = layer->input_norm = layer->stddev_inv = NULL;

    #pragma omp parallel for
    for (int j = 0; j < in_dim; j++) {
        layer->gamma->data[j] = 1.f;
    }

    randomize(layer->weights, 0.0f, sqrtf(2.0f / (float)in_dim));

    return layer;
}

void fc_free(fc_layer *layer) {
    
    matrix_free(layer->weights);
    matrix_free(layer->gamma);
    matrix_free(layer->beta);
    matrix_free(layer->run_mean);
    matrix_free(layer->run_var);

    if (layer->input != NULL){
        matrix_free(layer->input);
        matrix_free(layer->input_norm);
        matrix_free(layer->stddev_inv);
    }

    free(layer);
}

void fc_update_status(fc_layer *layer, matrix *mean, matrix *variance) {

    const float momentum = 0.9f;
    const float nmomentum = 1.0f - momentum;

    #pragma omp parallel for
    for (int i = 0; i < mean->columns; i++) {
        layer->run_mean->data[i] = (momentum * layer->run_mean->data[i]) + (nmomentum * mean->data[i]);
        layer->run_var->data[i] = (momentum * layer->run_var->data[i]) + (nmomentum * variance->data[i]);
    }
}

static inline void clear_cache(fc_layer *layer) {
    if (layer->input != NULL) {
        matrix_free(layer->input);
        matrix_free(layer->input_norm);
        matrix_free(layer->stddev_inv);
    }
}

static inline void train_forward(fc_layer *layer, matrix *raw_input) {
    
    matrix *mean, *var;

    mean = mean_0axis(raw_input);
    var = variance_0axis(raw_input, mean);

    fc_update_status(layer, mean, var);

    layer->stddev_inv = stddev_inv(var);
    layer->input_norm = normalized2(raw_input, mean, layer->stddev_inv);

    matrix_free(mean);
    matrix_free(var);
}

matrix* fc_forward(fc_layer *layer, matrix *raw_input, bool training) {

    clear_cache(layer);

    if (training) {
        train_forward(layer, raw_input);
    }
    
    else {
        layer->stddev_inv = stddev_inv(layer->run_var);
        layer->input_norm = normalized2(raw_input, layer->run_mean, layer->stddev_inv);
    }

    layer->input = elemwise_mulvec2(layer->input_norm, layer->gamma, layer->beta);

    matrix *output = multiply(layer->input, layer->weights, CblasNoTrans, CblasNoTrans, 
        layer->input->rows, layer->weights->columns, layer->input->columns);

    if (layer->relu) {
        relu(output);
    }

    return output;
}

static matrix* fc_bn_derivative(fc_layer *layer, matrix *dinput) {

    int rows = dinput->rows, columns = dinput->columns;
    float *dp1 = aligned_alloc(CACHE_LINE, sizeof(float) * columns * 2);
    float *dp2 = dp1 + columns;
    matrix *out = matrix_alloc(rows, columns);

    const float n_inv = 1.0f / (float)rows;

    #pragma omp parallel for
    for (int i = 0; i < 2 * columns; i++) {
        dp1[i] = .0f;
    }

    for (int i = 0; i < rows; i++) {
        register int index = i * columns;
        register float elem;
        for (int j = 0; j < columns; j++, index++) {
            elem = dinput->data[index] * layer->gamma->data[j];
            dp1[j] += elem;
            dp2[j] += elem * layer->input_norm->data[index];
            out->data[index] = elem * rows;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        register int index = i * columns;
        for (int j = 0; j < columns; j++, index++) {
            out->data[index] -= dp1[j] + (layer->input_norm->data[index] * dp2[j]);
            out->data[index] *= layer->stddev_inv->data[j] * n_inv;
        }
    }

    free(dp1);

    return out;
}

matrix* fc_backward(fc_layer *layer, matrix *dout, float lambda_reg, float l_rate) {

    // Current weights derivative with L2 regularization
    matrix *dweights = multiply(layer->input, dout, CblasTrans, CblasNoTrans, layer->input->columns, dout->columns, layer->input->rows);
    apply_sum(dweights, layer->weights, lambda_reg);

    // get in derivative
    matrix *dinput = multiply(dout, layer->weights, CblasNoTrans, CblasTrans, dout->rows, layer->weights->rows, dout->columns);

    // gamma and beta derivative for batch norm
    matrix *dgammap = elemwise_mul(dinput, layer->input_norm);
    matrix *dgamma = sum_rows(dgammap);
    matrix *dbeta = sum_rows(dinput);

    // get input layer final derivative
    matrix *dinput_norm = fc_bn_derivative(layer, dinput);

    // get relu derivative
    relu_del(dinput_norm, layer->input);

    // update weights
    apply_sum(layer->weights, dweights, (-1.0f) * l_rate);
    apply_sum(layer->gamma, dgamma, (-1.0f) * l_rate);
    apply_sum(layer->beta, dbeta, (-1.0f) * l_rate);

    // clear
    matrix_free(dinput);
    matrix_free(dbeta);
    matrix_free(dweights);
    matrix_free(dgamma);
    matrix_free(dgammap);

    return dinput_norm;
}
