//
// Created by vmachado on 2/11/20.
//

#ifndef matrix_h
#define matrix_h

#include <math.h>
#include <cblas.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <float.h>
#include "utils.h"

typedef struct _matrix {
    int rows, columns;
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

matrix *multiply(matrix *src, matrix *in, CBLAS_TRANSPOSE tra, CBLAS_TRANSPOSE trb, int m, int n, int k);

matrix *sum(matrix *src, matrix *in, float scalar);

matrix *elemwise_mul(matrix *src, matrix *in);

matrix *elemwise_mulvec2(matrix *src, matrix *in_1, matrix *in_2);

matrix *elemwise_mulvec(matrix *src, matrix *in);

matrix* mat_copy(matrix *src);

float sum_elem(matrix *src);

void normalize(matrix *src, matrix *mean, matrix *variance, int spatial, int channels);

void softmax(matrix *src);

void randomize(matrix *src, float mean, float deviation);

void set(matrix *src, matrix *W);

void set_array(matrix *src, float *data);

void set_row(matrix *src, float *data, int row_pos);

void accumulate(matrix *src, matrix *in);

void apply_sum(matrix *src, matrix *in, float scalar);

void apply_elw_mulvec(matrix *src, matrix* in);

void apply_elw_mulvec2(matrix *src, matrix* in_1, matrix *in_2);

matrix* stddev_inv(matrix *src);

void mcopy(float *dest, float *src, int len);

#endif //matrix_h
