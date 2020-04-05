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
    layer->gamma = matrix_alloc(1, in_c);
    layer->beta = matrix_alloc(1, in_c);
    layer->run_mean = matrix_alloc(1, in_c);
    layer->run_var = matrix_alloc(1, in_c);
    layer->input = layer->input_norm = layer->stddev_inv = layer->input_col = NULL;

    randomize(layer->filters, 0.0f, sqrtf(2.0f / (float)layer->out_c));
    for (int i = 0; i < in_c; i++) {
        layer->gamma->data[i] = 1.0f;
    }


    return layer;
}

void conv_free(conv_layer *layer) {
    matrix_free(layer->filters);
    matrix_free(layer->gamma);
    matrix_free(layer->beta);
    matrix_free(layer->run_mean);
    matrix_free(layer->run_var);

    if (layer->input != NULL){
        matrix_free(layer->input);
        matrix_free(layer->input_norm);
        matrix_free(layer->stddev_inv);
        matrix_free(layer->input_col);
    }

    free(layer);
}

static inline void conv_update_status(conv_layer *layer, matrix *mean, matrix *variance) {

    const float momentum = 0.9f;
    const float nmomentum = 1.0f - momentum;

    #pragma omp parallel for
    for (int i = 0; i < layer->in_c; i++) {
        layer->run_mean->data[i] = (momentum * layer->run_mean->data[i]) + (nmomentum * mean->data[i]);
        layer->run_var->data[i] = (momentum * layer->run_var->data[i]) + (nmomentum * variance->data[i]);
    }
}

static inline void clear_cache(conv_layer *layer) {
    if (layer->input != NULL) {
        matrix_free(layer->input);
        matrix_free(layer->input_norm);
        matrix_free(layer->stddev_inv);
        matrix_free(layer->input_col);
    }
}

static inline matrix* conv_mean(conv_layer *layer, matrix *src) {

    matrix *out = matrix_alloc(1, layer->in_c);
    int spatial = layer->in_w * layer->in_h;
    float n_inv = 1.0f / (float)(src->rows * spatial);

    for (int c = 0; c < layer->in_c; c++) {
        for (int b = 0; b < src->rows; b++) {
            register float *src_ptr = src->data + b * layer->in_dim + c * spatial;
            for (int i = 0; i < spatial; i++) {
                out->data[c] += src_ptr[i];
            }
        }
        out->data[c] *= n_inv;
    }
    return out;
}

static inline matrix* conv_var(conv_layer *layer, matrix *src, matrix *mean) {

    matrix *out = matrix_alloc(1, layer->in_c);
    int spatial = layer->in_w * layer->in_h;
    float n_inv = 1.0f / (float)(src->rows * spatial);

    for (int c = 0; c < layer->in_c; c++) {
        for (int b = 0; b < src->rows; b++) {
            register float *src_ptr = src->data + b * layer->in_dim + c * spatial;
            for (int i = 0; i < spatial; i++) {
                float diff = src_ptr[i] - mean->data[c];
                out->data[c] += diff * diff;
            }
        }
        out->data[c] *= n_inv;
    }
    return out;
}

static inline matrix* conv_norm(conv_layer *layer, matrix *src, matrix *mean, matrix *stddev_inv) {

    matrix *out = matrix_alloc(src->rows, src->columns);
    int spatial = layer->in_w * layer->in_h;

    #pragma omp parallel for
    for (int b = 0; b < src->rows; b++) {
        for (int c = 0; c < layer->in_c; c++) {
            register float *out_ptr = out->data + b * layer->in_dim + c * spatial;
            register float *src_ptr = src->data + b * layer->in_dim + c * spatial;
            for (int i = 0; i < spatial; i++) {
                out_ptr[i] = (src_ptr[i] - mean->data[c]) * stddev_inv->data[c];
            }
        }
    }
    return out;
}

static inline void train_forward(conv_layer *layer, matrix *raw_input) {
    
    matrix *mean, *var;

    mean = conv_mean(layer, raw_input);
    var = conv_var(layer, raw_input, mean);

    conv_update_status(layer, mean, var);

    layer->stddev_inv = stddev_inv(var);
    layer->input_norm = conv_norm(layer, raw_input, mean, layer->stddev_inv);

    matrix_free(mean);
    matrix_free(var);
}

static inline matrix* conv_scale_shift(conv_layer *layer) {

    matrix *out = matrix_alloc(layer->input_norm->rows, layer->input_norm->columns);
    int spatial = layer->in_w * layer->in_h;

    #pragma omp parallel for
    for (int b = 0; b < layer->input_norm->rows; b++) {
        for (int c = 0; c < layer->in_c; c++) {
            register float *out_ptr = out->data + b * layer->in_dim + c * spatial;
            register float *src_ptr = layer->input_norm->data + b * layer->in_dim + c * spatial;
            for (int i = 0; i < spatial; i++) {
                out_ptr[i] = (src_ptr[i] * layer->gamma->data[c]) + layer->beta->data[c];
            }
        }
    }

    return out;
}

static matrix* conv_bn_derivative(conv_layer *layer, matrix *dinput_norm) {

    int spatial = layer->in_w * layer->in_h;
    float *dp1 = aligned_alloc(CACHE_LINE, sizeof(float) * layer->in_c * 2);
    float *dp2 = dp1 + layer->in_c;
    matrix *out = matrix_alloc(dinput_norm->rows, dinput_norm->columns);

    const float n = (float)(dinput_norm->rows * spatial);
    const float n_inv = 1.0f / n;

    #pragma omp parallel for
    for (int i = 0; i < 2 * layer->in_c; i++) {
        dp1[i] = .0f;
    }

    for (int b = 0; b < dinput_norm->rows; b++) {
        for (int c = 0; c < layer->in_c; c++) {
            float *out_ptr = out->data + b * layer->in_dim + c * spatial;
            float *din_norm_ptr = dinput_norm->data + b * layer->in_dim + c * spatial;
            float *in_norm_ptr = layer->input_norm->data + b * layer->in_dim + c * spatial;
            float elem;
            for (int i = 0; i < spatial; i++) {
                elem = din_norm_ptr[i] * layer->gamma->data[c];
                dp1[c] += elem;
                dp2[c] += elem * in_norm_ptr[i];
                out_ptr[i] = elem * n;
            }
        }
    }

    #pragma omp parallel for
    for (int b = 0; b < dinput_norm->rows; b++) {
        for (int c = 0; c < layer->in_c; c++) {
            float *out_ptr = out->data + b * layer->in_dim + c * spatial;
            float *in_norm_ptr = layer->input_norm->data + b * layer->in_dim + c * spatial;
            for (int i = 0; i < spatial; i++) {
                out_ptr[i] -= (dp1[c] + (in_norm_ptr[i] * dp2[c]));
                out_ptr[i] *= layer->stddev_inv->data[c] * n_inv;
            }
        }
    }

    free(dp1);

    return out;
}

static inline void conv_update_weights(conv_layer *layer, matrix *dfilters, matrix *dgamma, matrix *dbeta, int batch_size, float l_rate) {

    int len = layer->out_c * layer->col_c;
    const float batch_inv = 1.0f / (float)batch_size;

    #pragma omp parallel for
    for (int f = 0; f < len; f++) {
        layer->filters->data[f] -= l_rate * (dfilters->data[f] * batch_inv);
    }

    #pragma omp parallel for
    for (int c = 0; c < layer->in_c; c++) {
        layer->gamma->data[c] -= l_rate * dgamma->data[c];
        layer->beta->data[c] -= l_rate * dbeta->data[c];
    }
}

static inline matrix* conv_sum_spatial(matrix *src, int spatial, int channels) {

    matrix *out = matrix_alloc(1, channels);

    for (int b = 0; b < src->rows; b++) {
        for (int c = 0; c < channels; c++) {
            register float *src_ptr = src->data + spatial * b * channels + c * spatial;
            for (int i = 0; i < spatial; i++) {
                out->data[c] += src_ptr[i];
            }
        }
    }

    return out;
}

static inline matrix* conv_gamma_del(conv_layer *layer, matrix *dinput_norm) {

    matrix *src = elemwise_mul(dinput_norm, layer->input_norm);
    matrix *out = conv_sum_spatial(src, layer->in_w * layer->in_h, layer->in_c) ;
    matrix_free(src);

    return out;
}

matrix* conv_forward(conv_layer *layer, matrix *raw_input, bool training) {
    
    clear_cache(layer);

    if (training) {
        train_forward(layer, raw_input);
    }
    else {
        layer->stddev_inv = stddev_inv(layer->run_var);
        layer->input_norm = conv_norm(layer, raw_input, layer->run_mean, layer->stddev_inv);
    }

    layer->input = conv_scale_shift(layer);

    matrix *out = matrix_alloc(raw_input->rows, layer->out_dim);
    layer->input_col = matrix_alloc(raw_input->rows, layer->col_dim);
    int out_cols = layer->col_w * layer->col_h;

    for (int i = 0; i < layer->input->rows; i++) {
        
        float *in_row = layer->input->data + i * layer->in_dim;
        float *col_row = layer->input_col->data + i * layer->col_dim;
        float *out_row = out->data + i * layer->out_dim;

        iam2cool(in_row, layer->in_c, layer->in_w, layer->in_h, layer->f_size, layer->stride, 
                layer->padd, layer->col_w, layer->col_h, layer->col_c, col_row);
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    layer->filters->rows, out_cols, layer->filters->columns, 
                    1.0f, layer->filters->data, layer->filters->columns, 
                    col_row, out_cols, 0.0f, out_row, out_cols);
    }

    relu(out);
    
    return out;
}

matrix* conv_backward(conv_layer *layer, matrix *dout, float l_rate) {

    assert(dout->rows == layer->input->rows);

    matrix *dfilters = matrix_alloc(layer->filters->rows, layer->filters->columns);
    matrix *dinput_norm = matrix_alloc(layer->input->rows, layer->input->columns);
    
    int spatial = layer->in_w * layer->in_h;
    int col_cols = layer->col_w * layer->col_h;
    float *dcol = aligned_alloc(CACHE_LINE, sizeof(float) * layer->col_dim);
    
    for (int i = 0; i < layer->input->rows; i++) {

        float *dout_row = dout->data + i * layer->out_dim;
        float *col_row = layer->input_col->data + i * layer->col_dim;
        float *din_norm_row = dinput_norm->data + i * layer->in_dim;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                    layer->out_c, layer->col_c, col_cols, 
                    1.0f, dout_row, col_cols, 
                    col_row, col_cols, 1.0f, dfilters->data, dfilters->columns);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    layer->filters->columns, col_cols, layer->filters->rows,
                    1.0f, layer->filters->data, layer->filters->columns, 
                    dout_row, col_cols, 0.0f, dcol, col_cols);

        cool2ami(dcol, layer->in_c, layer->in_w, layer->in_h, layer->f_size, layer->stride, 
                layer->padd, layer->col_w, layer->col_h, layer->col_c, din_norm_row);
    }

    matrix *dgamma = conv_gamma_del(layer, dinput_norm);
    matrix *dbeta = conv_sum_spatial(dinput_norm, spatial, layer->in_c);

    matrix *dinput = conv_bn_derivative(layer, dinput_norm);
    relu_del(dinput, layer->input);

    conv_update_weights(layer, dfilters, dgamma, dbeta, dinput->rows, l_rate);

    free(dcol);
    matrix_free(dfilters);
    matrix_free(dgamma);
    matrix_free(dbeta);
    matrix_free(dinput_norm);

    return dinput;
}