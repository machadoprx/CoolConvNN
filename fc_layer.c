//
// Created by vmachado on 2/11/20.
//

#include "fc_layer.h"

fc_layer* fc_alloc(int in_dim, int out_dim, bool relu) {

    fc_layer* layer = aligned_alloc(CACHE_LINE, sizeof(*layer));
    layer->weights = matrix_alloc(in_dim, out_dim);
    layer->gamma = matrix_alloc(1, out_dim);
    layer->beta = matrix_alloc(1, out_dim);
    layer->run_mean = matrix_alloc(1, out_dim);
    layer->run_var = matrix_alloc(1, out_dim);

    layer->relu = relu;
    layer->input = layer->out_norm = layer->stddev_inv = NULL;
    layer->activations = NULL;

    #pragma omp parallel for
    for (int j = 0; j < out_dim; j++) {
        layer->gamma->data[j] = 1.f;
    }

    randomize(layer->weights, 0.0f, sqrtf(2.0f / (float)in_dim));

    return layer;
}

static inline void clear_cache(fc_layer *layer) {
    if (layer->input != NULL){
        matrix_free(layer->input);
    }
    if (layer->out_norm != NULL) {
        free(layer->activations);
        matrix_free(layer->out_norm);
        matrix_free(layer->stddev_inv);
    }
}

void fc_free(fc_layer *layer) {
    matrix_free(layer->weights);
    matrix_free(layer->gamma);
    matrix_free(layer->beta);
    matrix_free(layer->run_mean);
    matrix_free(layer->run_var);
    clear_cache(layer);
    free(layer);
}

void fc_update_status(fc_layer *layer, matrix *mean, matrix *variance) {

    const float momentum = 0.99f;
    const float nmomentum = 1.0f - momentum;

    #pragma omp parallel for
    for (int i = 0; i < mean->columns; i++) {
        layer->run_mean->data[i] = (momentum * layer->run_mean->data[i]) + (nmomentum * mean->data[i]);
        layer->run_var->data[i] = (momentum * layer->run_var->data[i]) + (nmomentum * variance->data[i]);
    }
}

static inline void train_forward(fc_layer *layer, matrix *raw_out) {
    
    matrix *mean, *var;

    mean = mean_0axis(raw_out);
    var = variance_0axis(raw_out, mean);

    fc_update_status(layer, mean, var);

    layer->stddev_inv = stddev_inv(var);
    normalize2(raw_out, mean, layer->stddev_inv);
    layer->out_norm = mat_copy(raw_out);

    matrix_free(mean);
    matrix_free(var);
}

static void fc_bn_derivative(fc_layer *layer, matrix *dout) {

    int rows = dout->rows, columns = dout->columns;
    float *dp1 = aligned_alloc(CACHE_LINE, sizeof(float) * columns * 2);
    float *dp2 = dp1 + columns;

    const float n_inv = 1.0f / (float)rows;

    #pragma omp parallel for
    for (int i = 0; i < 2 * columns; i++) {
        dp1[i] = .0f;
    }

    for (int i = 0; i < rows; i++) {
        register int index = i * columns;
        for (int j = 0; j < columns; j++, index++) {
            dout->data[index] *= layer->gamma->data[j];
            dp1[j] += dout->data[index];
            dp2[j] += dout->data[index] * layer->out_norm->data[index];
            dout->data[index] *= rows;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        register int index = i * columns;
        for (int j = 0; j < columns; j++, index++) {
            dout->data[index] -= dp1[j] + (layer->out_norm->data[index] * dp2[j]);
            dout->data[index] *= layer->stddev_inv->data[j] * n_inv;
        }
    }

    free(dp1);
}

matrix* fc_forward(fc_layer *layer, matrix *raw_input, bool training) {

    clear_cache(layer);

    layer->input = mat_copy(raw_input);

    matrix *out = multiply(raw_input, layer->weights, CblasNoTrans, CblasNoTrans, 
                    raw_input->rows, layer->weights->columns, raw_input->columns);

    if (layer->relu) {

        if (training) {
            train_forward(layer, out);
        }
        
        else {
            layer->stddev_inv = stddev_inv(layer->run_var);
            normalize2(out, layer->run_mean, layer->stddev_inv);
            layer->out_norm = mat_copy(out);
        }

        apply_elw_mulvec2(out, layer->gamma, layer->beta);
        layer->activations = relu_activations(out);

    }

    return out;
}

matrix* fc_backward(fc_layer *layer, matrix *dout, float lambda_reg, float l_rate) {

    matrix *dgamma = NULL, *dbeta = NULL;

    if (layer->relu) {
        
        del_relu_activations(dout, layer->activations);

        matrix *dgammap = elemwise_mul(dout, layer->out_norm);
        dgamma = sum_rows(dgammap);
        dbeta = sum_rows(dout);

        fc_bn_derivative(layer, dout);

        matrix_free(dgammap);
    }

    matrix *dweights = multiply(layer->input, dout, CblasTrans, CblasNoTrans, layer->input->columns, dout->columns, layer->input->rows);
    apply_sum(dweights, layer->weights, lambda_reg);

    matrix *dinput = multiply(dout, layer->weights, CblasNoTrans, CblasTrans, dout->rows, layer->weights->rows, dout->columns);

    int scale = (-1.0f) * l_rate;
    apply_sum(layer->weights, dweights, scale);

    if (layer->relu) {
        apply_sum(layer->gamma, dgamma, scale);
        apply_sum(layer->beta, dbeta, scale);
        matrix_free(dgamma);
        matrix_free(dbeta);
    }

    matrix_free(dweights);

    return dinput;
}
