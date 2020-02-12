//
// Created by vmachado on 2/11/20.
//

#include "Matrix.h"

Matrix::Matrix(int rows, int columns) {

    this->rows = rows;
    this->columns = columns;
    this->data = new double[rows * columns];
    this->mean = nullptr;
    this->variance = nullptr;
    memset(this->data, 0, sizeof(double) * rows * columns);

}

Matrix::~Matrix() {
    delete []data;
    delete mean;
    delete variance;
}

Matrix* Matrix::transposed() {

    auto *R = new Matrix(columns, rows);

    if (columns == 1 || rows == 1) {
        std::memcpy(R->data, data, rows * columns * sizeof(double));
        return R;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            R->data[j * R->columns + i] = data[i * columns + j];
        }
    }

    return R;

}

Matrix* Matrix::normalized() {

    auto *R = new Matrix(rows, columns);

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

Matrix* Matrix::multiply(Matrix *W) {

    assert(columns == W->rows);

    auto *R = new Matrix(rows, W->columns);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, W->columns, columns, 1.0, data, columns, W->data, W->columns, 0, R->data, W->columns);

    return R;
}

Matrix* Matrix::sum(Matrix *W, double scalar) {

    assert((rows == W->rows) && (columns == W->columns));

    auto *R = new Matrix(rows, columns);

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < columns; j++) {
            int index = i * columns + j;
            R->data[index] = data[index] + (W->data[index] * scalar);
        }

    }

    return R;
}

Matrix* Matrix::hadamard(Matrix *W) {

    if (W->rows == 1) {
        assert(W->columns == columns);
    }
    else {
        assert((rows == W->rows) && (columns == W->columns));
    }

    auto *R = new Matrix(rows, columns);

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < columns; j++) {
            int index = i * columns + j;
            R->data[index] = data[index] * W->data[j];
        }

    }

    return R;
}

Matrix* Matrix::hadamard(Matrix *W, Matrix * W1) {

    assert((W->rows * W->columns) == columns);
    assert((W1->rows * W1->columns) == columns);

    auto *R = new Matrix(rows, columns);

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < columns; j++) {
            int index = i * columns + j;
            R->data[index] = (data[index] * W->data[j]) + W1->data[j];
        }

    }

    return R;
}

void Matrix::calcMean() {

    auto *T = transposed();
    delete mean;
    mean = new Matrix(1, columns);

    for (int i = 0; i < T->rows; i++) {

        double sum = 0;

        for (int j = 0; j < T->columns; j++) {
            sum += T->data[i * T->columns + j];
        }

        mean->data[i] = sum / (double) rows;
    }

    delete T;
}

void Matrix::calcVariance() {

    auto *T = transposed();
    delete variance;
    variance = new Matrix(1, columns);

    for (int i = 0; i < T->rows; i++) {

        double sum = 0;

        for (int j = 0; j < T->columns; j++) {
            sum += T->data[i * T->columns + j] * T->data[i * T->columns + j];
        }

        variance->data[i] = sum / (double) rows;
    }

    delete T;
}

Matrix* Matrix::centralized() {

    if (mean == nullptr) {
        calcMean();
    }

    auto *R = new Matrix(rows, columns);

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < columns; j++) {
            int index = i * columns + j;
            R->data[index] = data[index] - mean->data[j];
        }

    }

    return R;
}

Matrix* Matrix::centralized(Matrix *desiredMean) {

    auto *R = new Matrix(rows, columns);

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < columns; j++) {
            int index = i * columns + j;
            R->data[index] = data[index] - desiredMean->data[j];
        }

    }

    return R;
}

Matrix* Matrix::sumRows() {

    auto *R = new Matrix(1, columns);

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < columns; j++) {
            R->data[j] += data[i * columns + j];
        }

    }

    return R;
}

void Matrix::randomize() {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<double> distribution (0.0, sqrt(2.0 / rows));

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < columns; j++) {
            data[i * columns + j] = distribution(generator);
        }
    }
}

double Matrix::sumElements() {

    double sum = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            sum += data[i * columns + j];
        }
    }

    return sum;
}

Matrix* Matrix::invDeviation(Matrix *desiredVar, int len) {

    auto *R = new Matrix(1, len);
    double e = 0.00000001;

    for (int i = 0; i < len; i++) {
        R->data[i] = 1. / sqrt(desiredVar->data[i] + e);
    }

    return R;
}