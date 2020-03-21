//
// Created by vmachado on 2/11/20.
//

#include "Matrix.h"

Matrix::Matrix(int rows, int columns) {

    this->rows = rows;
    this->columns = columns;
    data = (float*) aligned_alloc(CACHE_LINE, sizeof(float) * rows * columns);
    
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            data[i] = .0f;
        }
    }
}

Matrix::~Matrix() {
    free(data);
}

Matrix* Matrix::transposed() {

    auto R = new Matrix(columns, rows);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {
            int index = i * columns;
            for (int j = 0; j < columns; j++) {
                R->data[j * R->columns + i] = data[index];
                index++;
            }
        }
    }

    return R;
}

Matrix* Matrix::normalized() {

    auto R = new Matrix(rows, columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {

            int row = i * columns, index = row, j;
            float sum = 0, max = -999999.0f;

            for (j = 0; j < columns; j++) {
                if (data[index] > max) {
                    max = data[index];
                }
                index++;
            }

            index = row;

            for (j = 0; j < columns; j++) {
                R->data[index] = expf(data[index] - max);
                sum += R->data[index];
                index++;
            }

            index = row;

            for (j = 0; j < columns; j++) {
                R->data[index] = R->data[index] / sum;
                index++;
            }
        }
    }

    return R;
}

void Matrix::normalize() {
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {

            int row = i * columns, index = row, j;
            float sum = 0, max = -999999.0f;

            for (j = 0; j < columns; j++) {
                if (data[index] > max) {
                    max = data[index];
                }
                index++;
            }

            index = row;

            for (j = 0; j < columns; j++) {
                data[index] = expf(data[index] - max);
                sum += data[index];
                index++;
            }

            index = row;

            for (j = 0; j < columns; j++) {
                data[index] = data[index] / sum;
                index++;
            }
        }
    }
}

Matrix* Matrix::multiply(Matrix* W) {
    auto R = new Matrix(rows, W->columns);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, W->columns, columns, 1.0, data, columns, W->data, W->columns, 0, R->data, R->columns);

    return R;
}

Matrix* Matrix::sum(Matrix* W, float scalar) {

    assert((rows == W->rows) && (columns == W->columns));

    auto R = new Matrix(rows, columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            R->data[i] = data[i] + (W->data[i] * scalar);
        }
    }

    return R;
}

void Matrix::apply_sum(Matrix* W, float scalar) {

    assert((rows == W->rows) && (columns == W->columns));

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            data[i] += (W->data[i] * scalar);
        }
    }
}

Matrix* Matrix::elemMul(Matrix* W) {

    assert((rows == W->rows) && (columns == W->columns));

    auto R = new Matrix(rows, columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            R->data[i] = data[i] * W->data[i];
        }
    }

    return R;
}

Matrix* Matrix::elemMulVector(Matrix* W, Matrix* W1) { // 1

    assert((W->rows * W->columns) == columns);
    assert((W1->rows * W1->columns) == columns);

    auto R = new Matrix(rows, columns);
    
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {
            int index = i * columns;
            for (int j = 0; j < columns; j++) {
                R->data[index] = (data[index] * W->data[j]) + W1->data[j];
                index++;
            }
        }
    }

    return R;
}

Matrix* Matrix::elemMulVector(Matrix* W) {

    assert((W->rows * W->columns) == columns);

    auto R = new Matrix(rows, columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {
            int index = i * columns;
            for (int j = 0; j < columns; j++) {
                R->data[index] = (data[index] * W->data[j]);
                index++;
            }
        }
    }

    return R;
}

Matrix* Matrix::mean0Axis() {

    auto T = transposed();
    auto mean = new Matrix(1, columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < columns; i++) {
    
            float sum = 0;
            int index = i * rows;
            
            for (int j = 0; j < rows; j++) {
                sum += T->data[index];
                index++;
            }

            mean->data[i] = sum / (float) rows;
        }
    }

    delete T;
    return mean;
}

Matrix* Matrix::variance0Axis(Matrix *mean) {

    auto T = transposed();
    auto variance = new Matrix(1, columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < columns; i++) {
            
            float sum = .0f, diff;
            int index = i * rows;

            for (int j = 0; j < rows; j++) {
                diff = T->data[index] - mean->data[i];
                sum += (diff * diff);
                index++;
            }

            variance->data[i] = sum / (float) rows;
        }
    }

    delete T;

    return variance;
}

Matrix* Matrix::normalized2(Matrix *mean, Matrix *deviationInv) {

    auto R = new Matrix(rows, columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {
            int index = i * columns;
            for (int j = 0; j < columns; j++) {
                R->data[index] = (data[index] - mean->data[j]) * deviationInv->data[j];
                index++;
            }
        }
    }

    return R;
}

void Matrix::normalize2(Matrix *mean, Matrix *deviationInv) {
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {
            int index = i * columns;
            for (int j = 0; j < columns; j++) {
                data[index] = (data[index] - mean->data[j]) * deviationInv->data[j];
                index++;
            }
        }
    }
}

Matrix* Matrix::sumRows() { //profile

    auto R = new Matrix(1, columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {
            int index = i * columns;
            for (int j = 0; j < columns; j++) {
                R->data[j] += data[index];
                index++;
            }
        }
    }

    return R;
}

Matrix* Matrix::sumColumns() { 

    auto R = new Matrix(1, rows);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {
            int index = i * columns;
            for (int j = 0; j < columns; j++) {
                R->data[i] += data[index];
                index++;
            }
        }
    }

    return R;
}

void Matrix::setRow(float *row, int rowPos) {

    float *ptr = data + (rowPos * columns);
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < columns; i++) {
            ptr[i] = row[i];
        }
    }
}

void Matrix::randomize(float mean, float deviation) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<float> distribution(mean, deviation);

    for (int i = 0; i < rows * columns; i++) {
        data[i] = distribution(generator);
    }
}

float Matrix::sumElements() {

    float sum = 0;

    #pragma omp parallel
    {
        #pragma omp for reduction (+:sum) schedule(static)
        for (int i = 0; i < rows * columns; i++) {
            sum += data[i];
        }
    }

    return sum;
}

Matrix* Matrix::invDeviation(Matrix* variance) {

    auto R = new Matrix(1, variance->columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < variance->columns; i++) {
            R->data[i] = 1.f / sqrtf(variance->data[i] + .000001f);
        }
    }

    return R;
}

Matrix* Matrix::ReLUDerivative(Matrix* W) { //profile

    assert(W->rows == rows && W->columns == columns);

    auto R = new Matrix(rows, columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            R->data[i] = (W->data[i] < .0f) ? .0f : data[i];
        }
    }

    return R;
}

void Matrix::apply_reluderivative(Matrix* W) { //profile

    assert(W->rows == rows && W->columns == columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            data[i] = (W->data[i] < .0f) ? .0f : data[i];
        }
    }
}

void Matrix::accumulate(Matrix *W) {

    assert(W->rows == rows && W->columns == columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            data[i] += W->data[i];
        }
    }
}

void Matrix::set(Matrix *W) {

    assert(W->rows == rows && W->columns == columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            data[i] = W->data[i];
        }
    }
}

void Matrix::setArray(float *data) {
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            this->data[i] = data[i];
        }
    }
}

Matrix* Matrix::copy() {

    auto R = new Matrix(rows, columns);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            R->data[i] = data[i];
        }
    }
    return R;
}

float Matrix::ReLU(float x) {
    return x > .0f ? x : .0f;
}

void Matrix::mcopy(float *dest, float *src, int size) {
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < size; i++) {
            dest[i] = src[i];
        }
    }
}