//
// Created by vmachado on 2/11/20.
//

#ifndef neural_net_h
#define neural_net_h

#include "fc_layer.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

matrix* correct_prob(matrix *prob, int *indices, int *labels);
matrix* prob_del(matrix* prob, int *indices, int *labels);
int* random_indices(int samples);

float loss(matrix* correct_prob);
float reg_loss(fc_layer **layers, int len, float l_reg);
void get_batch(int *indices, float **data_set, int batch_len, int data_dim, matrix* batch);

#endif //neural_net_h
