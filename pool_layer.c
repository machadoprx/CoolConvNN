#include "pool_layer.h"

pool_layer* pool_alloc(int chan, int in_w, int in_h, int f_size, int stride, int padd) {

    pool_layer *layer = aligned_alloc(CACHE_LINE, sizeof(*layer));

    layer->stride = stride;
    layer->f_size = f_size;  
    layer->padd = padd;
    layer->chan = chan;
    layer->in_w = in_w;
    layer->in_h = in_h;
    layer->out_w = 1 + (in_w + padd - f_size) / stride;
    layer->out_h = 1 + (in_h + padd - f_size) / stride;
    layer->out_dim = layer->out_w * layer->out_h * chan;
    layer->in_dim = in_h * in_w * chan;
    layer->indexes = NULL;

    return layer;
}

void pool_free(pool_layer *layer) {
    if (layer->indexes != NULL) {
        free(layer->indexes);
    }
    free(layer);
}

matrix* pool_forward(pool_layer *layer, matrix *raw_input) {

    if (layer->indexes != NULL) {
        free(layer->indexes);
    }

    int offset = -layer->padd / 2; // stride
    matrix *out = matrix_alloc(raw_input->rows, layer->out_dim);
    layer->indexes = aligned_alloc(CACHE_LINE, sizeof(int) * raw_input->rows * layer->out_dim);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < raw_input->rows; b++) {
        for (int c = 0; c < layer->chan; c++) {
            for(int h = 0; h < layer->out_h; h++) {
                for(int w = 0; w < layer->out_w; w++) {
                    int out_index = layer->out_w * ((layer->chan * b + c) * layer->out_h + h) + w;
                    int max_index = -1;
                    float max = -9999999.0f;
                    for (int fh = 0; fh < layer->f_size; fh++) {
                        for (int fw = 0; fw < layer->f_size; fw++) {
                            int curr_w = offset + (w * layer->stride) + fw;
                            int curr_h = offset + (h * layer->stride) + fh;
                            int in_index = curr_w + layer->in_w * (curr_h + layer->in_h * (c + b * layer->chan));
                            if (curr_h >= 0 && curr_h < layer->in_h && 
                                curr_w >= 0  && curr_w < layer->in_w) {
                                if (raw_input->data[in_index] > max) {
                                    max_index = in_index;
                                    max = raw_input->data[in_index];
                                }
                            }
                        }
                    }
                    out->data[out_index] = max;
                    layer->indexes[out_index] = max_index;
                }
            }
        }
    }

    return out;
}

matrix* pool_backward(pool_layer *layer, matrix *dout) {

    matrix *dinput = matrix_alloc(dout->rows, layer->in_dim);
    int size = layer->out_dim * dout->rows;

    for (int i = 0; i < size; ++i) {
        dinput->data[ layer->indexes[i] ] += dout->data[i];
    }

    return dinput;
}