//
// Created by vmachado on 2/11/20.
//

#include "neural_net.h"

matrix* correct_prob(matrix *prob, int *labels){

    matrix *correct_prob = matrix_alloc(prob->rows, 1);

    #pragma omp parallel for
    for (int i = 0; i < prob->rows; i++) {
        correct_prob->data[i] = (-1.0f) * logf(prob->data[i * prob->columns + labels[i]]);
    }

    return correct_prob;
}

matrix* prob_del(matrix* prob, int *labels){

    matrix *dprob = matrix_alloc(prob->rows, prob->columns);
    const float n_inv = 1.0f / (float)prob->rows;

    #pragma omp parallel for
    for (int i = 0; i < prob->rows; i++) {
        
        float *dp_row = dprob->data + i * prob->columns;
        float *p_row = prob->data + i * prob->columns;

        dp_row[ labels[i] ] = -1.0f;

        for (int j = 0; j < prob->columns; j++) {
            dp_row[j] = (dp_row[j] + p_row[j]) * n_inv;
        }
    }

    return dprob;
}

float loss(matrix* correct_prob){

    float out = .0f;

    #pragma omp parallel for reduction (+:out)
    for (int i = 0; i < correct_prob->rows; i++) {
        out += correct_prob->data[i];
    }

    return out / (float)correct_prob->rows;
}

float reg_loss(fc_layer **layers, int len, float l_reg){

    float out = .0f;

    for (int i = 0; i < len; i++) {
        matrix *w = layers[i]->weights;
        matrix *temp = elemwise_mul(w, w);
        out += 0.5f * l_reg * sum_elem(temp);
        matrix_free(temp);
    }

    return out;
}

int* random_indices(int samples) {

    int *indices = aligned_alloc(CACHE_LINE, sizeof(int) * samples);
    
    #pragma omp parallel for
    for (int i = 0; i < samples; i++) {
        indices[i] = i;
    }

    srand((int)time(0));

    for (int i = samples - 1; i >= 1; i--) {
        int rand_index = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[rand_index];
        indices[rand_index] = temp;
    }

    return indices;
}

void get_batch(int *indices, float **data_set, int *labels, int batch_len, int data_dim, matrix* batch, int *batch_labels) {
    
    #pragma omp parallel for
    for (int i = 0; i < batch_len; i++) {

        register float *dest_ptr = batch->data + (i * data_dim);
        for (int j = 0; j < data_dim; j++) {
            dest_ptr[j] = data_set[ indices[i] ][j];
        }
        batch_labels[i] = labels[ indices[i] ];
    }
}