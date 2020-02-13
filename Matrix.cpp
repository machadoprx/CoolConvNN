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

    auto *R = new Matrix(columns, rows);

    if (columns == 1 || rows == 1) {
        std::memcpy(R->data, data, rows * columns * sizeof(double));
        return R;
    }

    #pragma omp parallel
    #pragma omp for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            R->data[j * R->columns + i] = data[i * columns + j];
        }
    }

    return R;

}

Matrix* Matrix::normalized() {

    auto *R = new Matrix(rows, columns);

    #pragma omp parallel
    #pragma omp for
    for (int i = 0; i < rows; i++) {

        int index;
        double sum = 0, max = data[i * columns];

        for (int j = 1; j < columns; j++) {
            index = i * columns + j;
            if (data[index] > max) {
                max = data[index];
            }
        }

        for (int j = 0; j < columns; j++) {
            index = i * columns + j;
            R->data[index] = exp(data[index] - max);
            sum += R->data[index];
        }

        for (int j = 0; j < columns; j++) {
            index = i * columns + j;
            R->data[index] = R->data[index] / sum;
        }
    }

    return R;
}

Matrix* Matrix::multiply(Matrix* W) {

    assert(columns == W->rows);

    auto *R = new Matrix(rows, W->columns);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            rows, W->columns, columns, 1.0, data, columns, W->data, W->columns, 0, R->data, W->columns);

    /*auto *WT = W->transposed();
    for (int i = 0; i < R->rows; i++) {
        for (int j = 0; j < R->columns; j++) {
            int index = i * R->columns + j;
            for (int k = 0; k < columns; k++) {
                R->data[index] += data[i * columns + k] * WT->data[j * WT->columns + k]; //W1[i][k] * W2T[j][k];
            }
        }
    }
    delete WT;*/
    return R;
}

Matrix* Matrix::sum(Matrix* W, double scalar) {

    assert((rows == W->rows) && (columns == W->columns));

    auto *R = new Matrix(rows, columns);

    for (int i = 0; i < rows * columns; i++) {

        R->data[i] = data[i] + (W->data[i] * scalar);

    }

    return R;
}

Matrix* Matrix::elemMul(Matrix* W) {

    assert((rows == W->rows) && (columns == W->columns));

    auto *R = new Matrix(rows, columns);

    for (int i = 0; i < rows * columns; i++) {
        R->data[i] = data[i] * W->data[i];
    }

    return R;
}

Matrix* Matrix::elemMulVector(Matrix* W, Matrix* W1) {

    assert((W->rows * W->columns) == columns);
    assert((W1->rows * W1->columns) == columns);

    auto *R = new Matrix(rows, columns);

    for (int i = 0; i < rows * columns; i++) {

        int j = i % columns;
        R->data[i] = (data[i] * W->data[j]) + W1->data[j];

    }

    return R;
}

Matrix* Matrix::elemMulVector(Matrix* W) {

    assert((W->rows * W->columns) == columns);

    auto *R = new Matrix(rows, columns);

    for (int i = 0; i < rows * columns; i++) {

        int j = i % columns;
        R->data[i] = (data[i] * W->data[j]);

    }

    return R;
}

Matrix* Matrix::mean0Axis() {

    auto *T = transposed();
    auto *mean = new Matrix(1, columns);

    #pragma omp parallel
    #pragma omp for
    for (int i = 0; i < T->rows; i++) {

        double sum = 0;

        for (int j = 0; j < T->columns; j++) {
            sum += T->data[i * T->columns + j];
        }

        mean->data[i] = sum / (double) rows;
    }

    delete T;
    return mean;
}

Matrix* Matrix::variance0Axis() {

    auto *T = transposed();
    auto *variance = new Matrix(1, columns);

    #pragma omp parallel
    #pragma omp for
    for (int i = 0; i < T->rows; i++) {

        double sum = 0;

        for (int j = 0; j < T->columns; j++) {
            sum += T->data[i * T->columns + j] * T->data[i * T->columns + j];
        }

        variance->data[i] = sum / (double) rows;
    }

    delete T;

    return variance;
}

Matrix* Matrix::centralized(Matrix* desiredMean) {

    auto *R = new Matrix(rows, columns);

    for (int i = 0; i < rows * columns; i++) {

        int j = i % columns;
        R->data[i] = data[i] - desiredMean->data[j];

    }

    return R;
}

Matrix* Matrix::sumRows() {

    auto *R = new Matrix(1, columns);

    for (int i = 0; i < rows * columns; i++) {

        int j = i % columns;
        R->data[j] += data[i];

    }

    return R;
}

void Matrix::randomize() {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<double> distribution (0.0, sqrt(2.0 / rows));

    for (int i = 0; i < rows * columns; i++) {
        data[i] = distribution(generator);
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

    auto *R = new Matrix(1, desiredVar->columns);
    double e = 0.00000001;

    for (int i = 0; i < desiredVar->columns; i++) {
        R->data[i] = 1. / sqrt(desiredVar->data[i] + e);
    }

    return R;
}

Matrix* Matrix::copy() {

    auto *R = new Matrix(rows, columns);
    memcpy(R->data, data, sizeof(double) * rows * columns);

    return R;
}