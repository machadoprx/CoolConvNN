#ifndef activations_h
#define activations_h

#include "utils.h"
#include "matrix.h"

typedef struct _activations_layer{
    int type;
    matrix *cache;
} activations_layer;

activations_layer* activations_alloc(int type);
void activations_free(activations_layer *layer);

matrix* activations_forward(activations_layer *layer, matrix *input);
matrix* activations_backward(activations_layer *layer, matrix *dout);

float del_activate(float x, float x_old, int type);
float activate(float x, int type);

#endif
