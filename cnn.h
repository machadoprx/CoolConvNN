#ifndef cnn_h
#define cnn_h

#include "conv_layer.h"
#include "neural_net.h"
#include "pool_layer.h"
#include "fc_layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct _cnn {
    int conv_pool_layers;
    int fc_in, fc_out, fc_lsize, fc_add;
    int batch_size;
    float l_rate, l_reg;
    fc_layer **fc;
    conv_layer **conv;
    pool_layer **pool;
} cnn;

cnn* cnn_alloc(const char* cnn_config);
cnn* cnn_load(const char* cnn_config, const char* cnn_state);
void cnn_free(cnn *net);
void cnn_save(cnn *net, const char* cnn_state);
void cnn_train(cnn *net, float **data_set, int *labels, int samples, int epochs);

matrix* cnn_forward(cnn *net, matrix *batch, bool training);

#endif //cnn_h
