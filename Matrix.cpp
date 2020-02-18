//
// Created by vmachado on 2/11/20.
//

#include "Matrix.h"

Matrix::Matrix(int rows, int columns) {

    this->rows = rows;
    this->columns = columns;
    this->data = new double[rows * columns];
    memset(this->data, 0, sizeof(double) * rows * columns);

}

Matrix::~Matrix() {
    delete []data;
}

Matrix* Matrix::transposed() {

    auto R = new Matrix(columns, rows);

    if (columns == 1 || rows == 1) {
        std::memcpy(R->data, data, rows * columns * sizeof(double));
        return R;
    }

    int stage = rows / THREADS;

    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {
            int row = i * columns; 
            for (int j = 0; j < columns; j++) {
                R->data[j * R->columns + i] = data[row + j];
            }
        }
    }

    return R;

}

Matrix* Matrix::normalized() {

    auto R = new Matrix(rows, columns);

    int stage = rows / THREADS;

    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {

            int row = i * columns, index, j;
            double sum = 0, max = data[i * columns];

            for (j = 1; j < columns; j++) {
                index = row + j;
                if (data[index] > max) {
                    max = data[index];
                }
            }

            for (j = 0; j < columns; j++) {
                index = row + j;
                R->data[index] = exp(data[index] - max);
                sum += R->data[index];
            }

            for (j = 0; j < columns; j++) {
                index = row + j;
                R->data[index] = R->data[index] / sum;
            }
        }
    }

    return R;
}

Matrix* Matrix::multiply(Matrix* W) {

    assert(columns == W->rows);

    auto R = new Matrix(rows, W->columns);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            rows, W->columns, columns, 1.0, data, columns, W->data, W->columns, 0, R->data, W->columns);

    return R;
}

Matrix* Matrix::sum(Matrix* W, double scalar) {

    assert((rows == W->rows) && (columns == W->columns));

    auto R = new Matrix(rows, columns);
    memcpy(R->data, data, sizeof(double) * rows * columns);

    int stage = rows / THREADS;

    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {
            int row = i * columns;
            for (int j = 0; j < columns; j++) {
                int index = row + j;
                R->data[index] += W->data[index] * scalar;
            }
        }
    }

    return R;
}

Matrix* Matrix::elemMul(Matrix* W) {

    assert((rows == W->rows) && (columns == W->columns));

    auto R = new Matrix(rows, columns);


    int stage = rows / THREADS;

    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {
            int row = i * columns;
            for (int j = 0; j < columns; j++) {
                int index = row + j;
                R->data[index] = data[index] * W->data[index];
            }
        }
    }

    return R;
}

Matrix* Matrix::elemMulVector(Matrix* W, Matrix* W1) { // 1

    assert((W->rows * W->columns) == columns);
    assert((W1->rows * W1->columns) == columns);

    auto R = new Matrix(rows, columns);
    
    int stage = rows / THREADS;

    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {
            int row = i * columns;
            for (int j = 0; j < columns; j++) {
                int index = row + j;
                R->data[index] = (data[index] * W->data[j]) + W1->data[j];
            }
        }
    }

    return R;
}

Matrix* Matrix::elemMulVector(Matrix* W) {

    assert((W->rows * W->columns) == columns);

    auto R = new Matrix(rows, columns);


    int stage = rows / THREADS;

    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {
            int row = i * columns;
            for (int j = 0; j < columns; j++) {
                int index = row + j;
                R->data[index] = (data[index] * W->data[j]);
            }
        }
    }

    return R;
}

Matrix* Matrix::mean0Axis() {

    auto T = transposed();
    auto mean = new Matrix(1, columns);

    int stage = T->rows / THREADS;

    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {
            double sum = 0;
            int row = i * T->columns;
            for (int j = 0; j < T->columns; j++) {
                sum += T->data[row + j];
            }

            mean->data[i] = sum / (double) rows;
        }
    }

    delete T;
    return mean;
}

Matrix* Matrix::variance0Axis() {

    auto T = transposed();
    auto variance = new Matrix(1, columns);

    int stage = T->rows / THREADS;

    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {
            double sum = 0;
            int row = i * T->columns;
            for (int j = 0; j < T->columns; j++) {
                int index = row + j;
                sum += T->data[index] * T->data[index];
            }

            variance->data[i] = sum / (double) rows;
        }
    }

    delete T;

    return variance;
}

Matrix* Matrix::centralized(Matrix* desiredMean) {

    auto R = new Matrix(rows, columns);


    int stage = rows / THREADS;

    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {
            int row = i * columns;
            for (int j = 0; j < columns; j++) {
                int index = row + j;
                R->data[index] = data[index] - desiredMean->data[j];
            }
        }
    }

    return R;
}

Matrix* Matrix::sumRows() { //profile

    auto R = new Matrix(1, columns);

    int stage = rows / THREADS;

    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {
            int row = i * columns;
            for (int j = 0; j < columns; j++) {
                R->data[j] += data[row + j];
            }
        }
    }

    return R;
}

void Matrix::randomize() {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<double> distribution (0.0, sqrt(2.0 / rows));

    int stage = rows / THREADS;

    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {
            int row = i * columns;
            for (int j = 0; j < columns; j++) {
                data[row + j] = distribution(generator);
            }
        }
    }
}

double Matrix::sumElements() {

    double sum = 0;

    for (int i = 0; i < rows * columns; i++) {
        sum += data[i];
    }
    return sum;
}

Matrix* Matrix::invDeviation(Matrix* desiredVar) {

    auto R = new Matrix(1, desiredVar->columns);
    double e = 0.00000001;

    for (int i = 0; i < desiredVar->columns; i++) {
        R->data[i] = 1. / sqrt(desiredVar->data[i] + e);
    }

    return R;
}

Matrix* Matrix::ReLUDerivative(Matrix* W) { //profile

    assert(W->rows == rows && W->columns == columns);

    auto R = new Matrix(rows, columns);
    memcpy(R->data, data, sizeof(double) * rows * columns);


    int stage = rows / THREADS;

    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {
            int row = i * columns;
            for (int j = 0; j < columns; j++) {
                int index = row + j;
                if (W->data[index] < 0) {
                    R->data[index] = 0;
                }
            }
        }
    }

    return R;
}

void Matrix::set(Matrix *W) {

    assert(W->rows == rows && W->columns == columns);

    memcpy(data, W->data, rows * columns * sizeof(double));

}

Matrix* Matrix::copy() {

    auto R = new Matrix(rows, columns);
    memcpy(R->data, data, rows * columns * sizeof(double));

    return R;
}