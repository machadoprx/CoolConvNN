#include "bn_layer.h"

bn_layer *bn_alloc(int in_channels, int in_spatial) {
    
    bn_layer *layer = aalloc(sizeof(*layer));

    layer->in_channels = in_channels;
    layer->in_spatial = in_spatial;
    layer->gamma = matrix_alloc(1, layer->in_channels);
    layer->beta = matrix_alloc(1, layer->in_channels);
    layer->run_var = matrix_alloc(1, layer->in_channels);
    layer->run_mean = matrix_alloc(1, layer->in_channels);

    for (int i = 0; i < in_channels; i++) {
        layer->gamma->data[i] = 1.0f;
    }
    layer->variance = layer->out_cache = NULL;

    return layer;
}

static void clear_cache(bn_layer *layer) {
    matrix_free(layer->variance);
    matrix_free(layer->out_cache);
}

void bn_free(bn_layer *layer) {
    matrix_free(layer->gamma);
    matrix_free(layer->beta);
    matrix_free(layer->run_mean);
    matrix_free(layer->run_var);
    clear_cache(layer);
    free(layer);
}

static void bn_update_status(bn_layer *layer, matrix *mean, matrix *variance) {

    const float momentum = 0.99f;
    const float nmomentum = 1.0f - momentum;

    #pragma omp parallel for simd
    for (int i = 0; i < layer->in_channels; i++) {
        layer->run_mean->data[i] = (momentum * layer->run_mean->data[i]) + (nmomentum * mean->data[i]);
        layer->run_var->data[i] = (momentum * layer->run_var->data[i]) + (nmomentum * variance->data[i]);
    }
}

matrix* bn_forward(bn_layer *layer, matrix *input, bool training) {

    clear_cache(layer);

    if (training == true) {
        matrix *_mean;

        _mean = mean(input, layer->in_spatial, layer->in_channels);
        layer->variance = variance(input, _mean, layer->in_spatial, layer->in_channels);

        bn_update_status(layer, _mean, layer->variance);

        layer->out_cache = normalized(input, _mean, layer->variance, layer->in_spatial, layer->in_channels);

        matrix_free(_mean);
    }
    else {
        layer->out_cache = normalized(input, layer->run_mean, layer->run_var, layer->in_spatial, layer->in_channels);
    }

    return scale_shifted(layer->out_cache, layer->gamma, layer->beta, layer->in_channels, layer->in_spatial);
}

void bn_norm_del(matrix *dout, matrix *gamma, matrix *out_norm, matrix *variance, int spatial, int channels) {

    const float n = (float)(dout->rows * spatial);
    const float eps = 0.00001f;

    #pragma omp parallel for simd
    for (int c = 0; c < channels; c++) {
        register float dp1 = 0.0f, dp2 = 0.0f;
        const float _gamma = gamma->data[c];
        const float stddev_inv_n = 1.0f / (sqrtf(variance->data[c] + eps) * n);
        for (int b = 0; b < dout->rows; b++) {
            int index = spatial * (b * channels + c);
            float *dout_ptr = dout->data + index;
            float *out_norm_ptr = out_norm->data + index;
            for (int i = 0; i < spatial; i++) {
                dout_ptr[i] *= _gamma;
                dp1 += dout_ptr[i];
                dp2 += dout_ptr[i] * out_norm_ptr[i];
                dout_ptr[i] *= n;
            }
        }

        for (int b = 0; b < dout->rows; b++) {
            int index = spatial * (b * channels + c);
            float *dout_ptr = dout->data + index;
            float *out_norm_ptr = out_norm->data + index;
            for (int i = 0; i < spatial; i++) {
                dout_ptr[i] -= (dp1 + (out_norm_ptr[i] * dp2));
                dout_ptr[i] *= stddev_inv_n;
            }
        }
    }
}

static matrix* sum_spatial(matrix *src, int spatial, int channels) {

    matrix *out = matrix_alloc(1, channels);

    #pragma omp parallel for simd
    for (int c = 0; c < channels; c++) {
        for (int b = 0; b < src->rows; b++) {
            register float *src_ptr = src->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                out->data[c] += src_ptr[i];
            }
        }
    }

    return out;
}

matrix* bn_backward(bn_layer *layer, matrix *dout, float l_rate) {
    
    matrix *out = mat_copy(dout);
    // scale shift del
    matrix *dp = elemwise_mul(out, layer->out_cache);
    matrix *dgamma = sum_spatial(dp, layer->in_spatial, layer->in_channels);
    matrix *dbeta = sum_spatial(out, layer->in_spatial, layer->in_channels);

    //norm del
    bn_norm_del(out, layer->gamma, layer->out_cache, layer->variance, layer->in_spatial, layer->in_channels);
    
    apply_sum(layer->gamma, dgamma, -l_rate);
    apply_sum(layer->beta, dbeta, -l_rate);
    
    matrix_free(dp);
    matrix_free(dgamma);
    matrix_free(dbeta);
    return out;
}