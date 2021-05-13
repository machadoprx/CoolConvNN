//
// Created by vmachado on 2/11/20.
//

#ifndef matrix_h
#define matrix_h

#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include "gemm.h"
#include "utils.h"

typedef struct _matrix {
    unsigned rows, columns;
    float *data;
} matrix;

matrix* matrix_alloc(int rows, int columns);

void matrix_free(matrix *src);

matrix* transposed(matrix *src);

matrix *softmaxed(matrix *src);

matrix *normalized(matrix *src, matrix *mean, matrix *variance, int spatial, int channels);

matrix *sum_rows(matrix *src);

matrix *sum_columns(matrix *src);

matrix* scale_shifted(matrix *src, matrix *gamma, matrix *beta, int channels, int spatial);

matrix *variance(matrix *src, matrix *mean, int spatial, int channels);

matrix* mean(matrix *src, int spatial, int channels);

matrix *multiply(matrix *src, matrix *in, bool tra, bool trb, int m, int n, int k);

matrix *sum(matrix *src, matrix *in, float scalar);

matrix *elemwise_mul(matrix *src, matrix *in);

matrix* mat_copy(matrix *src);

float sum_elem(matrix *src);

void normalize(matrix *src, matrix *mean, matrix *variance, int spatial, int channels);

void softmax(matrix *src);

void randomize(matrix *src, float mean, float deviation);

void apply_sum(matrix *src, matrix *in, float scalar);

#endif //matrix_h
