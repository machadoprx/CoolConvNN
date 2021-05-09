#include "conv_layer.h"

conv_layer* conv_alloc(int in_c, int in_w, int in_h, int out_c, int f_size, int stride, int padd) {
    
    conv_layer *layer = aalloc(sizeof(*layer));

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

    layer->input_col = NULL;

    randomize(layer->filters, 0.0f, sqrtf(2.0f / (float)layer->in_dim));

    return layer;
}

void conv_free(conv_layer *layer) {
    matrix_free(layer->filters);
    matrix_free(layer->input_col);
    free(layer);
}

matrix* conv_forward(conv_layer *layer, matrix *raw_input) {
    
    matrix_free(layer->input_col);

    matrix *out = matrix_alloc(raw_input->rows, layer->out_dim);
    layer->input_col = matrix_alloc(raw_input->rows, layer->col_dim);
    int out_cols = layer->col_w * layer->col_h;
    matrix *tmp = matrix_alloc(layer->filters->rows, out_cols)
    for (int i = 0; i < raw_input->rows; i++) {
        
        float *in_row = raw_input->data + i * layer->in_dim;
        float *col_row = layer->input_col->data + i * layer->col_dim;
        float *out_row = out->data + i * layer->out_dim;

        iam2cool(in_row, layer->in_c, layer->in_w, layer->in_h, layer->f_size, layer->stride, 
                layer->padd, layer->col_w, layer->col_h, layer->col_c, col_row);
        matrix *mul = multiply(layer->filters, matrix *in, bool tra, bool trb, int m, int n, int k)
        /*cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    layer->filters->rows, out_cols, layer->filters->columns, 
                    1.0f, layer->filters->data, layer->filters->columns, 
                    col_row, out_cols, 0.0f, out_row, out_cols);*/
    }

    return out;
}

matrix* conv_backward(conv_layer *layer, matrix *dout, float l_rate) {

    assert(dout->rows == layer->input_col->rows);

    int batch_size = layer->input_col->rows;
    int spatial = layer->col_w * layer->col_h;

    float *dcol = aalloc(sizeof(float) * layer->col_dim);
    matrix *dfilters = matrix_alloc(layer->filters->rows, layer->filters->columns);
    matrix *dinput = matrix_alloc(batch_size, layer->in_dim);

    for (int i = 0; i < batch_size; i++) {

        float *dout_row = dout->data + i * layer->out_dim;
        float *col_row = layer->input_col->data + i * layer->col_dim;
        float *din_row = dinput->data + i * layer->in_dim;

        /*cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                    layer->out_c, layer->col_c, spatial, 
                    1.0f, dout_row, spatial, 
                    col_row, spatial, 1.0f, dfilters->data, dfilters->columns);*/

        /*cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    layer->filters->columns, spatial, layer->filters->rows,
                    1.0f, layer->filters->data, layer->filters->columns, 
                    dout_row, spatial, 0.0f, dcol, spatial);*/

        cool2ami(dcol, layer->in_c, layer->in_w, layer->in_h, layer->f_size, layer->stride, 
                layer->padd, layer->col_w, layer->col_h, layer->col_c, din_row);
    }

    apply_sum(layer->filters, dfilters, -l_rate / (float)batch_size);
    
    free(dcol);
    matrix_free(dfilters);

    return dinput;
}
