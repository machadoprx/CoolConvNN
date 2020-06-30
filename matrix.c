//
// Created by vmachado on 2/11/20.
//

#include "matrix.h"

matrix* matrix_alloc(int rows, int columns) {

    int len = rows * columns;
    matrix *out = aalloc(sizeof(*out));
    out->data = aalloc(sizeof(float) * len);

    out->rows = rows;
    out->columns = columns;
    
    for (int i = 0; i < len; i++) {
        out->data[i] = .0f;
    }

    return out;
}

static inline matrix *internal_alloc(int rows, int columns) {
    matrix *out = aalloc(sizeof(*out));
    out->data = aalloc(sizeof(float) * rows * columns);
    out->rows = rows;
    out->columns = columns;

    return out;
}

void matrix_free(matrix *src) {
    if (src != NULL) {
        free(src->data);
        free(src);
    }
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

matrix* softmaxed(matrix *src) {

    matrix *out = internal_alloc(src->rows, src->columns);

    #pragma omp parallel for
    for (int i = 0; i < src->rows; i++) {

        int j;
        float sum = 0, max = -FLT_MAX;
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

void softmax(matrix *src) {
    #pragma omp parallel for
    for (int i = 0; i < src->rows; i++) {

        int j;
        float sum = 0, max = -FLT_MAX;
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

    for (int i = 0; i < len; i++) {
        out->data[i] = src->data[i] + (in->data[i] * scalar);
    }

    return out;
}

void apply_sum(matrix *src, matrix *in, float scalar) {

    assert((src->rows == in->rows) && (src->columns == in->columns));
    int len = src->rows * src->columns;

    for (int i = 0; i < len; i++) {
        src->data[i] += (in->data[i] * scalar);
    }
}

matrix* elemwise_mul(matrix *src, matrix* in) {

    assert((src->rows == in->rows) && (src->columns == in->columns));

    int len = src->rows * src->columns;
    matrix *out = internal_alloc(src->rows, src->columns);

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

matrix* mean(matrix *src, int spatial, int channels) {

    matrix *out = matrix_alloc(1, channels);
    const float n_inv = 1.0f / (float)(src->rows * spatial);

    #pragma omp parallel for
    for (int c = 0; c < channels; c++) {
        for (int b = 0; b < src->rows; b++) {
            register float *src_ptr = src->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                out->data[c] += src_ptr[i];
            }
        }
        out->data[c] *= n_inv;
    }

    return out;
}

matrix* variance(matrix *src, matrix *mean, int spatial, int channels) {
    matrix *out = matrix_alloc(1, channels);
    const float n_inv = 1.0f / (float)(src->rows * spatial);

    #pragma omp parallel for
    for (int c = 0; c < channels; c++) {
        for (int b = 0; b < src->rows; b++) {
            register float *src_ptr = src->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                float diff = src_ptr[i] - mean->data[c];
                out->data[c] += diff * diff;
            }
        }
        out->data[c] *= n_inv;
    }
    return out;
}

void normalize(matrix *src, matrix *mean, matrix *variance, int spatial, int channels) {

    const float eps = 1e-5f;
    #pragma omp parallel for
    for (int b = 0; b < src->rows; b++) {
        for (int c = 0; c < channels; c++) {
            register float *src_ptr = src->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                src_ptr[i] = (src_ptr[i] - mean->data[c]) / sqrtf(variance->data[c] + eps);
            }
        }
    }
}

matrix* normalized(matrix *src, matrix *mean, matrix *variance, int spatial, int channels) {
    
    matrix *out = matrix_alloc(src->rows, src->columns);
    
    const float eps = 1e-5f;
    #pragma omp parallel for
    for (int b = 0; b < src->rows; b++) {
        for (int c = 0; c < channels; c++) {
            register float *src_ptr = src->data + spatial * (b * channels + c);
            register float *out_ptr = out->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                out_ptr[i] = (src_ptr[i] - mean->data[c]) / sqrtf(variance->data[c] + eps);
            }
        }
    }

    return out;
}

matrix* scale_shifted(matrix *src, matrix *gamma, matrix *beta, int channels, int spatial) {
    
    matrix *out = matrix_alloc(src->rows, src->columns);
    
    #pragma omp parallel for
    for (int b = 0; b < src->rows; b++) {
        for (int c = 0; c < channels; c++) {
            register float *src_ptr = src->data + spatial * (b * channels + c);
            register float *out_ptr = out->data + spatial * (b * channels + c);
            for (int i = 0; i < spatial; i++) {
                out_ptr[i] = (src_ptr[i] * gamma->data[c]) + beta->data[c];
            }
        }
    }
    return out;
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

void accumulate(matrix* src, matrix *in) {

    assert(in->rows == src->rows && in->columns == src->columns);

    int len = src->rows * src->columns;

    for (int i = 0; i < len; i++) {
        src->data[i] += in->data[i];
    }
}

void set(matrix *src, matrix *in) {

    assert(in->rows == src->rows && in->columns == src->columns);

    int len = src->rows * src->columns;

    for (int i = 0; i < len; i++) {
        src->data[i] = in->data[i];
    }
}

void set_array(matrix *src, float *data) {

    int len = src->rows * src->columns;

    for (int i = 0; i < len; i++) {
        src->data[i] = data[i];
    }
}

matrix* mat_copy(matrix *src) {

    matrix *out = internal_alloc(src->rows, src->columns);
    int len = src->rows * src->columns;

    for (int i = 0; i < len; i++) {
        out->data[i] = src->data[i];
    }

    return out;
}

void mcopy(float *dest, float *src, int len) {
    for (int i = 0; i < len; i++) {
        dest[i] = src[i];
    }
}