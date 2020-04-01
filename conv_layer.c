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

    layer->filters = matrix_alloc(layer->out_c, layer->col_c);
    layer->bias = matrix_alloc(1, layer->out_c);
    layer->input = NULL;

    randomize(layer->filters, 0.0f, sqrtf(2.0f / (float)layer->out_c));

    return layer;
}

void conv_free(conv_layer *layer) {
    matrix_free(layer->filters);
    matrix_free(layer->bias);
    if (layer->input != NULL) 
        matrix_free(layer->input);
    free(layer);
}

static void conv_bias_relu(conv_layer *layer, matrix *conv) {
    
    int size = layer->col_w * layer->col_h;

    #pragma omp parallel for
    for (int b = 0; b < conv->rows; b++) {
        for (int i = 0; i < layer->out_c; i++) {
            float *conv_row = conv->data + size * (b * layer->out_c + i);
            for (int j = 0; j < size; j++) {
                conv_row[j] += layer->bias->data[i];
                if (conv_row[j] < .0f) {
                    conv_row[j] = .0f;
                }
            }
        }
    }
}

static void conv_update_weights(conv_layer *layer, matrix *dfilters, matrix *dbias, float l_rate, int batch_size) {
    
    float batch_inv = 1.0f / (float)batch_size;

    #pragma omp parallel for
    for (int i = 0; i < layer->out_c; i++) {
        float *filter_row = layer->filters->data + i * layer->col_c;
        float *df_row = dfilters->data + i * layer->col_c;
        for (int j = 0; j < layer->col_c; j++) {
            filter_row[j] -= (df_row[j] * batch_inv) * l_rate;
        }
        layer->bias->data[i] -= (dbias->data[i] * batch_inv) * l_rate;
    }
}

matrix* conv_forward(conv_layer *layer, matrix *raw_input) {

    if (layer->input != NULL) {
        matrix_free(layer->input);
    }

    matrix *out = matrix_alloc(raw_input->rows, layer->out_dim);
    layer->input = matrix_alloc(raw_input->rows, layer->col_dim);
    int out_cols = layer->col_w * layer->col_h;

    for (int i = 0; i < raw_input->rows; i++) {
        
        float *raw_row = raw_input->data + i * layer->in_dim;
        float *col_row = layer->input->data + i * layer->col_dim;
        float *out_row = out->data + i * layer->out_dim;

        iam2cool(raw_row, layer->in_c, layer->in_w, layer->in_h, layer->f_size, layer->stride, 
                layer->padd, layer->col_w, layer->col_h, layer->col_c, col_row);
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    layer->filters->rows, out_cols, layer->filters->columns, 
                    1.0f, layer->filters->data, layer->filters->columns, 
                    col_row, out_cols, 0.0f, out_row, out_cols);
    }

    conv_bias_relu(layer, out);
    return out;
}

matrix* conv_backward(conv_layer *layer, matrix *dout, float l_rate) {

    assert(dout->rows == layer->input->rows);

    matrix *dinput = matrix_alloc(layer->input->rows, layer->in_dim);
    matrix *dfilters = matrix_alloc(layer->filters->rows, layer->filters->columns);
    matrix *dbias =  matrix_alloc(layer->bias->rows, layer->bias->columns);

    for (int i = 0; i < layer->input->rows; i++) {

        matrix *col = matrix_alloc(layer->col_c, layer->col_w * layer->col_h);
        matrix *dout_row = matrix_alloc(layer->out_c, layer->col_w * layer->col_h);

        mcopy(col->data, layer->input->data + (i * layer->col_dim), layer->col_dim);
        mcopy(dout_row->data, dout->data + (i * layer->out_dim), layer->out_dim);

        matrix *df_row = multiply(dout_row, col, CblasNoTrans, CblasTrans, dout_row->rows, col->rows, dout_row->columns);
        matrix *dbias_row = sum_columns(dout_row);

        accumulate(dfilters, df_row);
        accumulate(dbias, dbias_row);

        matrix *dinput_row = multiply(layer->filters, dout_row, CblasTrans, CblasNoTrans, layer->filters->columns, dout_row->columns, layer->filters->rows);
        relu_del(dinput_row, col);

        cool2ami(dinput_row->data, layer->in_c, layer->in_w, layer->in_h, layer->f_size, layer->stride, 
                layer->padd, layer->col_w, layer->col_h, layer->col_c, dinput->data + i * dinput->columns);
        
        matrix_free(df_row);
        matrix_free(dbias_row);
        matrix_free(col);
        matrix_free(dinput_row);
        matrix_free(dout_row);
    }

    conv_update_weights(layer, dfilters, dbias, l_rate, dinput->rows);
    matrix_free(dfilters);
    matrix_free(dbias);

    return dinput;
}