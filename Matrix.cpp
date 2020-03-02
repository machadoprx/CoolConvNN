//
// Created by vmachado on 2/11/20.
//

#include "Matrix.h"

Matrix::Matrix(int rows, int columns) {

    this->rows = rows;
    this->columns = columns;
    data = (float*) aligned_alloc(CACHE_LINE, sizeof(float) * rows * columns);
    
    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            data[i] = 0;
        }
    }
}

Matrix::~Matrix() {
    free(data);
}

Matrix* Matrix::transposed() {

    auto R = new Matrix(columns, rows);

    #pragma omp parallel num_threads(THREADS)
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

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {

            int row = i * columns, index = row;
            float sum = 0, max = data[index];

            for (int j = 0; j < columns; j++) {
                if (data[index] > max) {
                    max = data[index];
                }
                index++;
            }

            index = row;

            for (int j = 0; j < columns; j++) {
                R->data[index] = exp(data[index] - max);
                sum += R->data[index];
                index++;
            }

            index = row;

            for (int j = 0; j < columns; j++) {
                R->data[index] = R->data[index] / sum;
                index++;
            }
        }
    }

    return R;
}

Matrix* Matrix::multiply(Matrix* W) {

    assert(columns == W->rows);

    auto R = new Matrix(rows, W->columns);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, W->columns, columns, 1.0, data, columns, W->data, W->columns, 0, R->data, R->columns);

    return R;
}

Matrix* Matrix::sum(Matrix* W, float scalar) {

    assert((rows == W->rows) && (columns == W->columns));

    auto R = new Matrix(rows, columns);

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            R->data[i] = data[i] + (W->data[i] * scalar);
        }
    }

    return R;
}

Matrix* Matrix::elemMul(Matrix* W) {

    assert((rows == W->rows) && (columns == W->columns));

    auto R = new Matrix(rows, columns);

    #pragma omp parallel num_threads(THREADS)
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
    
    #pragma omp parallel num_threads(THREADS)
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

    #pragma omp parallel num_threads(THREADS)
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

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < T->rows; i++) {
    
            float sum = 0;
            int index = i * T->columns;
            
            for (int j = 0; j < T->columns; j++) {
                sum += T->data[index];
                index++;
            }

            mean->data[i] = sum / (float) rows;
        }
    }

    delete T;
    return mean;
}

Matrix* Matrix::variance0Axis() {

    auto T = transposed();
    auto variance = new Matrix(1, columns);

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < T->rows; i++) {
            
            float sum = 0;
            int index = i * T->columns;

            for (int j = 0; j < T->columns; j++) {
                sum += T->data[index] * T->data[index];
                index++;
            }
            variance->data[i] = sum / (float) rows;
        }
    }

    delete T;

    return variance;
}

Matrix* Matrix::centralized(Matrix* desiredMean) {

    auto R = new Matrix(rows, columns);

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {
            int index = i * columns;
            for (int j = 0; j < columns; j++) {
                R->data[index] = data[index] - desiredMean->data[j];
                index++;
            }
        }
    }

    return R;
}

Matrix* Matrix::sumRows() { //profile

    auto R = new Matrix(1, columns);

    #pragma omp parallel num_threads(THREADS)
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

void Matrix::randomize() {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<float> distribution(0.0, sqrt(2.0 / rows));

    for (int i = 0; i < rows * columns; i++) {
        data[i] = distribution(generator);
    }
}

float Matrix::sumElements() {

    float sum = 0;

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait reduction (+:sum)
        for (int i = 0; i < rows * columns; i++) {
            sum += data[i];
        }
    }

    return sum;
}

Matrix* Matrix::invDeviation(Matrix* desiredVar) {

    auto R = new Matrix(1, desiredVar->columns);
    float e = .00001f;

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < desiredVar->columns; i++) {
            R->data[i] = 1.0f / sqrt(desiredVar->data[i] + e);
        }
    }

    return R;
}

Matrix* Matrix::ReLUDerivative(Matrix* W) { //profile

    assert(W->rows == rows && W->columns == columns);

    auto R = new Matrix(rows, columns);

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            if (W->data[i] < 0) {
                R->data[i] = 0;
            }
            else {
                R->data[i] = data[i];
            }
        }
    }

    return R;
}

void Matrix::set(Matrix *W) {

    assert(W->rows == rows && W->columns == columns);

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            data[i] = W->data[i];
        }
    }
}

Matrix* Matrix::copy() {

    auto R = new Matrix(rows, columns);

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < rows * columns; i++) {
            R->data[i] = data[i];
        }
    }
    return R;
}

float Matrix::ReLU(float x) {
    return x > 0 ? x : 0;
}