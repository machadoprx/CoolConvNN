//
// Created by vmachado on 2/11/20.
//

#include "neural_net.h"

float accurracy(matrix *prob, int *indices, int *labels) {
    int correct = 0;
    const float threshold = 0.5f;
    for (int i = 0; i < prob->rows; i++) {
        if (prob->data[i * prob->columns + labels[ indices[i] ]] > threshold) {
            correct++;
        }
    }

    return (float)correct / (float)prob->rows;
}

matrix* prob_del(matrix* prob, int *indices, int *labels) {

    matrix *dprob = matrix_alloc(prob->rows, prob->columns);
    const float n_inv = 1.0f / (float)prob->rows;

    #pragma omp parallel for
    for (int i = 0; i < prob->rows; i++) {
        
        float *dp_row = dprob->data + i * prob->columns;
        float *p_row = prob->data + i * prob->columns;

        dp_row[ labels[ indices[i] ] ] = -1.0f;
        for (int j = 0; j < prob->columns; j++) {
            dp_row[j] = (dp_row[j] + p_row[j]) * n_inv;
        }
    }

    return dprob;
}

float loss(matrix* prob, int *indices, int *labels){

    float out = .0f;

    #pragma omp parallel for reduction(+: out)
    for (int i = 0; i < prob->rows; i++) {
        out += logf(prob->data[i * prob->columns + labels[ indices[i] ]]);
    }

    return -out / (float)prob->rows;
}

float reg_loss(void **layers, int *layer_type, int len, float l_reg) {

    float out = .0f;

    for (int i = 0; i < len; i++) {
        if (layer_type[i] == FC) {
            matrix *w = ((fc_layer*)layers[i])->weights;
            matrix *temp = elemwise_mul(w, w);
            out += 0.5f * l_reg * sum_elem(temp);
            matrix_free(temp);
        }
    }

    return out;
}

int* random_indices(int samples) {

    int *indices = aalloc(sizeof(int) * samples);
    
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

matrix* get_batch(int *indices, float **data_set, int batch_len, int data_dim) {
    
    matrix *batch = matrix_alloc(batch_len, data_dim);

    #pragma omp parallel for
    for (int i = 0; i < batch_len; i++) {

        register float *dest_ptr = batch->data + (i * data_dim);
        for (int j = 0; j < data_dim; j++) {
            dest_ptr[j] = data_set[ indices[i] ][j];
        }
    }
    return batch;
}