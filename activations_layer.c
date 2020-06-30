#include "activations_layer.h"

activations_layer* activations_alloc(int type) {
    activations_layer *layer = aalloc(sizeof(*layer));
    layer->cache = NULL;
    layer->type = type;
    return layer;
}

void activations_free(activations_layer *layer) {
    matrix_free(layer->cache);
    free(layer);
}

matrix* activations_forward(activations_layer *layer, matrix *input) {
    
    matrix_free(layer->cache);

    matrix *out = matrix_alloc(input->rows, input->columns);
    
    int len = input->rows * input->columns;
    if (layer->type == RELU) {
        layer->cache = mat_copy(input);
    }

    for (int i = 0; i < len; i++) {
        out->data[i] = activate(input->data[i], layer->type);
    }
    return out;
}

matrix* activations_backward(activations_layer *layer, matrix *dout) {
    matrix *out = matrix_alloc(dout->rows, dout->columns);
    
    int len = dout->rows * dout->columns;
    if (layer->type == RELU) {
        for (int i = 0; i < len; i++) {
            out->data[i] = del_activate(dout->data[i], layer->cache->data[i], layer->type);
        }
    }
    else {
        for (int i = 0; i < len; i++) {
            out->data[i] = del_activate(dout->data[i], 0, layer->type);
        }
    }

    return out;
}

float _relu(float x) {
    return x > 0 ? x : 0;
}

float _del_relu(float x, float old_x) {
    return old_x > 0 ? x : 0;
}

float activate(float x, int type) {
    if (type == RELU) {
        return _relu(x);
    }
    return -1;
}

float del_activate(float x, float x_old, int type) {
    if (type == RELU) {
        return _del_relu(x, x_old);
    }
    return -1;
}