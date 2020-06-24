#ifndef cool_nn_h
#define cool_nn_h

#include "conv_layer.h"
#include "neural_net.h"
#include "pool_layer.h"
#include "fc_layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct _cool_nn {
    int layers_n;
    int batch_size;
    int *layers_type;
    void **layers;
} cool_nn;

cool_nn* cool_alloc(const char* nn_config);
cool_nn* cool_load(const char* nn_config, const char* nn_state);

void cool_free(cool_nn *net);
void cool_save(cool_nn *net, const char* nn_state);
void cool_train(cool_nn *net, float **data_set, int *labels, int samples, float val_split, float l_rate, float l_reg, int epochs);

matrix* cool_forward(cool_nn *net, matrix *batch, bool training);

#endif //cool_nn_h
