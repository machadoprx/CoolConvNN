//
// Created by vmachado on 2/11/20.
//

#include "matrix.h"

matrix* matrix_alloc(int rows, int columns) {

    matrix *out = aligned_alloc(CACHE_LINE, sizeof(*out));
    out->data = aligned_alloc(CACHE_LINE, sizeof(float) * rows * columns);

    out->rows = rows;
    out->columns = columns;
    
    #pragma omp parallel for
    for (int i = 0; i < rows * columns; i++) {
        out->data[i] = .0f;
    }

    return out;
}

static inline matrix *internal_alloc(int rows, int columns) {
    matrix *out = aligned_alloc(CACHE_LINE, sizeof(*out));
    out->data = aligned_alloc(CACHE_LINE, sizeof(float) * rows * columns);
    out->rows = rows;
    out->columns = columns;

    return out;
}

void matrix_free(matrix *src) {
    free(src->data);
    free(src);
}

matrix* transposed(matrix *src) {

    matrix *out = internal_alloc(src->columns, src->rows);

    #pragma omp parallel for
    for (int i = 0; i < src->rows; i++) {
        register float *src_row = src->data + i * src->columns;
        for (int j = 0; j < src->columns; j++) {
            out->data[j * src->rows + i] = src_row[j];
        }
    }

    return out;
}

matrix* normalized(matrix *src) {

    matrix *out = internal_alloc(src->rows, src->columns);

    #pragma omp parallel for
    for (int i = 0; i < src->rows; i++) {

        int j;
        float sum = 0, max = -999999.0f;
        register float *src_row = src->data + i * src->columns;
        register float *out_row = out->data + i * src->columns;

        for (j = 0; j < src->columns; j++) {
            if (src_row[j] > max) {
                max = src_row[j];
            }
        }

        for (j = 0; j < src->columns; j++) {
            out_row[j] = expf(src_row[j] - max);
            sum += out_row[j];
        }

        float inv_sum = 1.0f / sum;

        for (j = 0; j < src->columns; j++) {
            out_row[j] *= inv_sum;
        }
    }

    return out;
}

void normalize(matrix *src) {
    #pragma omp parallel for
    for (int i = 0; i < src->rows; i++) {

        int j;
        float sum = 0, max = -999999.0f;
        register float *src_row = src->data + i * src->columns;

        for (j = 0; j < src->columns; j++) {
            if (src_row[j] > max) {
                max = src_row[j];
            }
        }

        for (j = 0; j < src->columns; j++) {
            src_row[j] = expf(src_row[j] - max);
            sum += src_row[j];
        }

        float inv_sum = 1.0f / sum;

        for (j = 0; j < src->columns; j++) {
            src_row[j] *= inv_sum;
        }
    }
}

matrix *multiply(matrix *src, matrix *in, CBLAS_TRANSPOSE tra, CBLAS_TRANSPOSE trb, int m, int n, int k) {

    matrix *out = internal_alloc(m, n);
    cblas_sgemm(CblasRowMajor, tra, trb,
        m, n, k, 1.0f, src->data, src->columns, in->data, in->columns, 0.0f, out->data, out->columns);

    return out;
}

matrix* sum(matrix *src, matrix *in, float scalar) {

    assert((src->rows == in->rows) && (src->columns == in->columns));

    matrix *out = internal_alloc(src->rows, src->columns);
    int len = src->rows * src->columns;

    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        out->data[i] = src->data[i] + (in->data[i] * scalar);
    }

    return out;
}

void apply_sum(matrix *src, matrix *in, float scalar) {

    assert((src->rows == in->rows) && (src->columns == in->columns));
    int len = src->rows * src->columns;

    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        src->data[i] += (in->data[i] * scalar);
    }
}

matrix* elemwise_mul(matrix *src, matrix* in) {

    assert((src->rows == in->rows) && (src->columns == in->columns));

    int len = src->rows * src->columns;
    matrix *out = internal_alloc(src->rows, src->columns);

    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        out->data[i] = src->data[i] * in->data[i];
    }

    return out;
}

matrix* elemwise_mulvec2(matrix *src, matrix* in_1, matrix* in_2) {

    assert((in_1->rows * in_1->columns) == src->columns);
    assert((in_2->rows * in_2->columns) == src->columns);

    matrix *out = internal_alloc(src->rows, src->columns);
    
    #pragma omp parallel for
    for (int i = 0; i < src->rows; i++) {
        float *src_row = src->data + i * src->columns;
        float *out_row = out->data + i * src->columns;
        for (int j = 0; j < src->columns; j++) {
            out_row[j] = (src_row[j] * in_1->data[j]) + in_2->data[j];
        }
    }

    return out;
}

void apply_elw_mulvec(matrix *src, matrix* in) {

    assert((in->rows * in->columns) == src->columns);
    
    #pragma omp parallel for
    for (int i = 0; i < src->rows; i++) {
        register float *src_row = src->data + i * src->columns;
        for (int j = 0; j < src->columns; j++) {
            src_row[j] *= in->data[j];
        }
    }
}

void apply_elw_mulvec2(matrix *src, matrix* in_1, matrix *in_2) {

    assert((in_1->rows * in_1->columns) == src->columns);
    assert((in_2->rows * in_2->columns) == src->columns);
    
    #pragma omp parallel for
    for (int i = 0; i < src->rows; i++) {
        register float *src_row = src->data + i * src->columns;
        for (int j = 0; j < src->columns; j++) {
            src_row[j] = (src_row[j] * in_1->data[j]) + in_2->data[j];
        }
    }
}

matrix* elemwise_mulvec(matrix *src, matrix* in) {

    assert((in->rows * in->columns) == src->columns);

    matrix *out = internal_alloc(src->rows, src->columns);
    
    #pragma omp parallel for
    for (int i = 0; i < src->rows; i++) {
        register float *src_row = src->data + i * src->columns;
        register float *out_row = out->data + i * src->columns;
        for (int j = 0; j < src->columns; j++) {
            out_row[j] = src_row[j] * in->data[j];
        }
    }

    return out;
}

matrix* mean_0axis(matrix *src) {

    matrix *out = matrix_alloc(1, src->columns);
    float inv_rows = 1.0f / (float)src->rows;

    for (int i = 0; i < src->rows; i++) {

        register float* src_row = src->data + i * src->columns;
        
        for (int j = 0; j < src->columns; j++) {
            out->data[j] += src_row[j];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < src->columns; i++) {
        out->data[i] *= inv_rows;
    }

    return out;
}

matrix* variance_0axis(matrix *src, matrix *mean) {

    matrix *out = matrix_alloc(1, src->columns);
    float inv_rows = 1.0f / (float)src->rows;

    for (int i = 0; i < src->rows; i++) {

        register float* src_row = src->data + i * src->columns;
        
        for (int j = 0; j < src->columns; j++) {
            float diff = src_row[j] - mean->data[j];
            out->data[j] += diff * diff;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < src->columns; i++) {
        out->data[i] *= inv_rows;
    }

    return out;
}

matrix* normalized2(matrix *src, matrix *mean, matrix *stddev_inv) {

    matrix *out = internal_alloc(src->rows, src->columns);

    #pragma omp parallel for
    for (int i = 0; i < src->rows; i++) {
        float *out_row = out->data + i * src->columns;
        float *src_row = src->data + i * src->columns;
        for (int j = 0; j < src->columns; j++) {
            out_row[j] = (src_row[j] - mean->data[j]) * stddev_inv->data[j];
        }
    }

    return out;
}

void normalize2(matrix *src, matrix *mean, matrix *stddev_inv) {
    
    #pragma omp parallel for
    for (int i = 0; i < src->rows; i++) {
        register float *src_row = src->data + i * src->columns;
        for (int j = 0; j < src->columns; j++) {
            src_row[j] = (src_row[j] - mean->data[j]) * stddev_inv->data[j];
        }
    }
}

matrix* sum_rows(matrix *src) { //profile

    matrix *out = matrix_alloc(1, src->columns);

    for (int i = 0; i < src->rows; i++) {

        register float* src_row = src->data + i * src->columns;
        
        for (int j = 0; j < src->columns; j++) {
            out->data[j] += src_row[j];
        }
    }

    return out;
}

matrix* sum_columns(matrix *src) { //profile

    matrix *out = matrix_alloc(1, src->rows);

    #pragma omp parallel for
    for (int i = 0; i < src->rows; i++) {

        register float* src_row = src->data + i * src->columns;
        
        for (int j = 0; j < src->columns; j++) {
            out->data[i] += src_row[j];
        }
    }

    return out;
}

void set_row(matrix *src, float *in, int row_pos) {
    
    float *src_ptr = src->data + (row_pos * src->columns);

    #pragma omp parallel for
    for (int i = 0; i < src->columns; i++) {
        src_ptr[i] = in[i];
    }
}

void randomize(matrix *src, float mean, float stddev) {
    
    srand((unsigned)time(0));
    int len = src->rows * src->columns;
    int range = len + len % 2;
    float u, v;

    for (int i = 0; i < range; i += 2) {    
        get_gauss(mean, stddev, &u, &v);
        src->data[i] = u;
        src->data[i + 1] = v;
    }
}

float sum_elem(matrix *src) {

    float sum = 0;
    int len = src->rows * src->columns;

    #pragma omp simd reduction(+: sum)
    for (int i = 0; i < len; i++) {
        sum += src->data[i];
    }

    return sum;
}

matrix* stddev_inv(matrix* src) {

    matrix *out = internal_alloc(1, src->columns);
    const float eps = 1e-5f;
    
    #pragma omp parallel for
    for (int i = 0; i < src->columns; i++) {
        out->data[i] = 1.0f / sqrtf(src->data[i] + eps);
    }

    return out;
}

int *relu_activations(matrix* src) {

    int len = src->rows * src->columns;
    int *atv = aligned_alloc(CACHE_LINE, len * sizeof(int));

    for (int i = 0; i < len; i++) {
        if (src->data[i] < .0f) {
            src->data[i] = .0f;
            atv[i] = 0;
        }
        else {
            atv[i] = 1;
        }
    }

    return atv;
}

void del_relu_activations(matrix* src, int* atv) {

    int len = src->rows * src->columns;

    for (int i = 0; i < len; i++) {
        src->data[i] = src->data[i] * (float)atv[i]; 
    }
}

void accumulate(matrix* src, matrix *in) {

    assert(in->rows == src->rows && in->columns == src->columns);

    int len = src->rows * src->columns;

    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        src->data[i] += in->data[i];
    }
}

void set(matrix *src, matrix *in) {

    assert(in->rows == src->rows && in->columns == src->columns);

    int len = src->rows * src->columns;

    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        src->data[i] = in->data[i];
    }
}

void set_array(matrix *src, float *data) {

    int len = src->rows * src->columns;

    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        src->data[i] = data[i];
    }
}

matrix* mat_copy(matrix *src) {

    matrix *out = internal_alloc(src->rows, src->columns);
    int len = src->rows * src->columns;

    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        out->data[i] = src->data[i];
    }

    return out;
}

void mcopy(float *dest, float *src, int len) {
    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        dest[i] = src[i];
    }
}