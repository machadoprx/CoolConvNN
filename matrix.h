//
// Created by vmachado on 2/11/20.
//

#ifndef matrix_h
#define matrix_h

#include <math.h>
#include <mkl/mkl.h>
#include <mkl/mkl_cblas.h>
#include <mkl/mkl_types.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

typedef struct _matrix {
    int rows, columns;
    float *data;
} matrix;

matrix* matrix_alloc(int rows, int columns);

void matrix_free(matrix *src);

matrix* transposed(matrix *src);

matrix *normalized(matrix *src);

matrix *normalized2(matrix *src, matrix *mean, matrix *stddev_inv);

matrix *sum_rows(matrix *src);

matrix *sum_columns(matrix *src);

matrix *variance_0axis(matrix *src, matrix *mean);

matrix* mean_0axis(matrix *src);

matrix *multiply(matrix *src, matrix *in, CBLAS_TRANSPOSE tra, CBLAS_TRANSPOSE trb, int m, int n, int k);

matrix *sum(matrix *src, matrix *in, float scalar);

matrix *elemwise_mul(matrix *src, matrix *in);

matrix *elemwise_mulvec2(matrix *src, matrix *in_1, matrix *in_2);

matrix *elemwise_mulvec(matrix *src, matrix *in);

matrix* mat_copy(matrix *src);

float sum_elem(matrix *src);

void normalize2(matrix *src, matrix *mean, matrix *stddev_inv);

void normalize(matrix *src);

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

int *relu_activations(matrix* src);

void del_relu_activations(matrix* src, int* atv) ;

#endif //matrix_h
