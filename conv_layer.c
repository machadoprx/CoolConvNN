#include "conv_layer.h"

conv_layer* conv_alloc(int in_c, int in_w, int in_h, int out_c, int f_size, int stride, int padd) {
    
    conv_layer *layer = aligned_alloc(CACHE_LINE, sizeof(*layer));

    layer->out_c = out_c; 
    layer->stride = stride;
    layer->f_size = f_size; 
    layer->in_c = in_c; 
    layer->padd = padd;
    layer->in_w = in_w;
    layer->in_h = in_h;
    layer->col_w = ((in_w - f_size + (2 * padd)) / stride) + 1;
    layer->col_h = ((in_h - f_size + (2 * padd)) / stride) + 1;
    layer->in_dim = in_w * in_h * in_c;
    layer->out_dim = layer->col_w * layer->col_h * out_c;
    layer->col_c = in_c * f_size * f_size;
    layer->col_dim = layer->col_c * layer->col_w * layer->col_h;

    layer->filters = matrix_alloc(out_c, layer->col_c);
    layer->gamma = matrix_alloc(1, out_c);
    layer->beta = matrix_alloc(1, out_c);
    layer->run_mean = matrix_alloc(1, out_c);
    layer->run_var = matrix_alloc(1, out_c);

    layer->out = layer->out_norm = layer->stddev_inv = layer->input_col = NULL;
    layer->activations = NULL;

    randomize(layer->filters, 0.0f, sqrtf(2.0f / (float)layer->out_c));
    for (int i = 0; i < out_c; i++) {
        layer->gamma->data[i] = 1.0f;
    }


    return layer;
}

static inline void clear_cache(conv_layer *layer) {
    if (layer->input_col != NULL){
        free(layer->activations);
        matrix_free(layer->out);
        matrix_free(layer->out_norm);
        matrix_free(layer->stddev_inv);
        matrix_free(layer->input_col);
    }
}

void conv_free(conv_layer *layer) {
    matrix_free(layer->filters);
    matrix_free(layer->gamma);
    matrix_free(layer->beta);
    matrix_free(layer->run_mean);
    matrix_free(layer->run_var);
    clear_cache(layer);
    free(layer);
}

static inline void conv_update_status(conv_layer *layer, matrix *mean, matrix *variance, int channels) {

    const float momentum = 0.99f;
    const float nmomentum = 1.0f - momentum;

    #pragma omp parallel for
    for (int i = 0; i < channels; i++) {
        layer->run_mean->data[i] = (momentum * layer->run_mean->data[i]) + (nmomentum * mean->data[i]);
        layer->run_var->data[i] = (momentum * layer->run_var->data[i]) + (nmomentum * variance->data[i]);
    }
}

static inline matrix* conv_mean(matrix *src, int spatial, int channels) {

    matrix *out = matrix_alloc(1, channels);
    const float n_inv = 1.0f / (float)(src->rows * spatial);

    for (int c = 0; c < channels; c++) {
        for (int b = 0; b < src->rows; b++) {
            register float *src_ptr = src->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                out->data[c] += src_ptr[i];
            }
        }
        out->data[c] *= n_inv;
    }
    return out;
}

static inline matrix* conv_var(matrix *src, matrix *mean, int spatial, int channels) {

    matrix *out = matrix_alloc(1, channels);
    const float n_inv = 1.0f / (float)(src->rows * spatial);

    for (int c = 0; c < channels; c++) {
        for (int b = 0; b < src->rows; b++) {
            register float *src_ptr = src->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                float diff = src_ptr[i] - mean->data[c];
                out->data[c] += diff * diff;
            }
        }
        out->data[c] *= n_inv;
    }
    return out;
}

static inline matrix* conv_norm(matrix *src, matrix *mean, matrix *stddev_inv, int spatial, int channels) {

    matrix *out = matrix_alloc(src->rows, src->columns);

    #pragma omp parallel for
    for (int b = 0; b < src->rows; b++) {
        for (int c = 0; c < channels; c++) {
            register float *out_ptr = out->data + spatial * (b * channels + c);
            register float *src_ptr = src->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                out_ptr[i] = (src_ptr[i] - mean->data[c]) * stddev_inv->data[c];
            }
        }
    }
    return out;
}

static inline void train_forward(conv_layer *layer, matrix *raw_out, int spatial, int channels) {
    
    matrix *mean, *var;

    mean = conv_mean(raw_out, spatial, channels);
    var = conv_var(raw_out, mean, spatial, channels);

    conv_update_status(layer, mean, var, channels);

    layer->stddev_inv = stddev_inv(var);
    layer->out_norm = conv_norm(raw_out, mean, layer->stddev_inv, spatial, channels);

    matrix_free(mean);
    matrix_free(var);
}

static inline matrix* conv_scale_shift(conv_layer *layer, matrix *src, int spatial, int channels) {

    matrix *out = matrix_alloc(src->rows, src->columns);

    #pragma omp parallel for
    for (int b = 0; b < src->rows; b++) {
        for (int c = 0; c < channels; c++) {
            register float *out_ptr = out->data + spatial * (b * channels + c);
            register float *src_ptr = src->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                out_ptr[i] = (src_ptr[i] * layer->gamma->data[c]) + layer->beta->data[c];
            }
        }
    }

    return out;
}

static void conv_bn_derivative(conv_layer *layer, matrix *dout, int spatial, int channels) {

    float *dp1 = aligned_alloc(CACHE_LINE, sizeof(float) * channels * 2);
    float *dp2 = dp1 + channels;

    const float n = (float)(dout->rows * spatial);
    const float n_inv = 1.0f / n;

    #pragma omp parallel for
    for (int i = 0; i < 2 * channels; i++) {
        dp1[i] = .0f;
    }

    for (int b = 0; b < dout->rows; b++) {
        for (int c = 0; c < channels; c++) {
            float *dout_ptr = dout->data + spatial * (b * channels + c);
            float *out_norm_ptr = layer->out_norm->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                dout_ptr[i] *= layer->gamma->data[c];
                dp1[c] += dout_ptr[i];
                dp2[c] += dout_ptr[i] * out_norm_ptr[i];
                dout_ptr[i] *= n;
            }
        }
    }

    #pragma omp parallel for
    for (int b = 0; b < dout->rows; b++) {
        for (int c = 0; c < channels; c++) {
            float *dout_ptr = dout->data + spatial * (b * channels + c);
            float *out_norm_ptr = layer->out_norm->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                dout_ptr[i] -= (dp1[c] + (out_norm_ptr[i] * dp2[c]));
                dout_ptr[i] *= layer->stddev_inv->data[c] * n_inv;
            }
        }
    }

    free(dp1);
}

static inline void conv_update_filters(conv_layer *layer, matrix *dfilters, int batch_size, float l_rate) {

    const int len = dfilters->rows * dfilters->columns;
    const float batch_inv = 1.0f / (float)batch_size;

    #pragma omp parallel for
    for (int f = 0; f < len; f++) {
        layer->filters->data[f] -= (dfilters->data[f] * batch_inv) * l_rate;
    }
}

static inline void conv_update_bn(conv_layer *layer, matrix *dgamma, matrix *dbeta, float l_rate) {
    #pragma omp parallel for
    for (int c = 0; c < dgamma->columns; c++) {
        layer->gamma->data[c] -= dgamma->data[c] * l_rate;
        layer->beta->data[c] -= dbeta->data[c] * l_rate;
    }
}

static inline matrix* conv_sum_spatial(matrix *src, int spatial, int channels) {

    matrix *out = matrix_alloc(1, channels);

    for (int b = 0; b < src->rows; b++) {
        for (int c = 0; c < channels; c++) {
            register float *src_ptr = src->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                out->data[c] += src_ptr[i];
            }
        }
    }

    return out;
}

static inline matrix* conv_gamma_del(conv_layer *layer, matrix *dout_norm, int spatial, int channels) {

    matrix *src = elemwise_mul(dout_norm, layer->out_norm);
    matrix *out = conv_sum_spatial(src, spatial, channels);
    matrix_free(src);

    return out;
}

matrix* conv_forward(conv_layer *layer, matrix *raw_input, bool training) {
    
    clear_cache(layer);

    matrix *out = matrix_alloc(raw_input->rows, layer->out_dim);
    layer->input_col = matrix_alloc(raw_input->rows, layer->col_dim);
    int out_cols = layer->col_w * layer->col_h;

    for (int i = 0; i < raw_input->rows; i++) {
        
        float *in_row = raw_input->data + i * layer->in_dim;
        float *col_row = layer->input_col->data + i * layer->col_dim;
        float *out_row = out->data + i * layer->out_dim;

        iam2cool(in_row, layer->in_c, layer->in_w, layer->in_h, layer->f_size, layer->stride, 
                layer->padd, layer->col_w, layer->col_h, layer->col_c, col_row);
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    layer->filters->rows, out_cols, layer->filters->columns, 
                    1.0f, layer->filters->data, layer->filters->columns, 
                    col_row, out_cols, 0.0f, out_row, out_cols);
    }
    
    if (training) {
        train_forward(layer, out, out_cols, layer->out_c);
    }
    else {
        layer->stddev_inv = stddev_inv(layer->run_var);
        layer->out_norm = conv_norm(out, layer->run_mean, layer->stddev_inv, out_cols, layer->out_c);
    }
    
    layer->out = conv_scale_shift(layer, layer->out_norm, out_cols, layer->out_c);
    layer->activations = relu_activations(layer->out);
    
    mcopy(out->data, layer->out->data, out->columns * out->rows);

    return out;
}

matrix* conv_backward(conv_layer *layer, matrix *dout, float l_rate) {

    assert(dout->rows == layer->input_col->rows);

    int batch_size = layer->input_col->rows;
    int spatial = layer->col_w * layer->col_h;

    del_relu_activations(dout, layer->activations);

    matrix *dgamma = conv_gamma_del(layer, dout, spatial, layer->out_c);
    matrix *dbeta = conv_sum_spatial(dout, spatial, layer->out_c);
    conv_bn_derivative(layer, dout, spatial, layer->out_c);
    
    conv_update_bn(layer, dgamma, dbeta, l_rate);
    
    matrix_free(dgamma);
    matrix_free(dbeta);

    float *dcol = aligned_alloc(CACHE_LINE, sizeof(float) * layer->col_dim);
    matrix *dfilters = matrix_alloc(layer->filters->rows, layer->filters->columns);
    matrix *dinput = matrix_alloc(batch_size, layer->in_dim);

    for (int i = 0; i < batch_size; i++) {

        float *dout_row = dout->data + i * layer->out_dim;
        float *col_row = layer->input_col->data + i * layer->col_dim;
        float *din_row = dinput->data + i * layer->in_dim;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                    layer->out_c, layer->col_c, spatial, 
                    1.0f, dout_row, spatial, 
                    col_row, spatial, 1.0f, dfilters->data, dfilters->columns);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    layer->filters->columns, spatial, layer->filters->rows,
                    1.0f, layer->filters->data, layer->filters->columns, 
                    dout_row, spatial, 0.0f, dcol, spatial);

        cool2ami(dcol, layer->in_c, layer->in_w, layer->in_h, layer->f_size, layer->stride, 
                layer->padd, layer->col_w, layer->col_h, layer->col_c, din_row);
    }

    conv_update_filters(layer, dfilters, batch_size, l_rate);
    
    free(dcol);
    matrix_free(dfilters);

    return dinput;
}