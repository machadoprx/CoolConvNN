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

    Matrix(int rows, int columns);

    ~Matrix();

    Matrix* transposed();
    
    Matrix* normalized();
    
    Matrix* sumRows();
    
    Matrix* variance0Axis();

    Matrix* mean0Axis();

    Matrix *multiply(Matrix *W);

    Matrix *sum(Matrix *W, double scalar);

    Matrix *elemMul(Matrix *W);

    Matrix *elemMulVector(Matrix *W, Matrix *W1);

    Matrix *elemMulVector(Matrix *W);

    Matrix *centralized(Matrix *desiredMean);

    Matrix *copy();

    double sumElements();

    void randomize();

    static Matrix *invDeviation(Matrix *desiredVar);
};


#endif //NNCPP_EXEC_MATRIX_H
