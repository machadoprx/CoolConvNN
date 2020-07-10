//
// Created by vmachado on 2/11/20.
//

#ifndef neural_net_h
#define neural_net_h

#include "fc_layer.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

matrix* prob_del(matrix* prob, int *indices, int *labels);
int* random_indices(int samples);

float loss(matrix* prob, int *indices, int *labels);
float reg_loss(void **layers, int *layer_type, int len, float l_reg);
float accurracy(matrix *prob, int *indices, int *labels);
matrix* get_batch(int *indices, float **data_set, int batch_len, int data_dim);

#endif //neural_net_h
