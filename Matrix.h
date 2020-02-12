//
// Created by vmachado on 2/11/20.
//

#ifndef NNCPP_EXEC_MATRIX_H
#define NNCPP_EXEC_MATRIX_H


#include <cstring>
#include <array>
#include <cmath>
#include <cblas.h>
#include <cassert>
#include <random>
#include <chrono>

class Matrix {

private:

public:
    int rows{}, columns{};
    double *data{};
    Matrix *mean{};
    Matrix *variance{};

    Matrix(int rows, int columns);
    ~Matrix();

    Matrix* transposed();
    Matrix* normalized();
    Matrix* centralized();
    Matrix* centralized(Matrix*);
    Matrix* sumRows();
    static Matrix* invDeviation(Matrix*, int);

    Matrix* multiply(Matrix*);
    Matrix* sum(Matrix*, double);
    Matrix* hadamard(Matrix*);
    Matrix* hadamard(Matrix*, Matrix*);

    double sumElements();

    void randomize();
    void calcMean();
    void calcVariance();
};


#endif //NNCPP_EXEC_MATRIX_H
