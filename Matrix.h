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
#include <iterator>

#define THREADS 8
#define CACHE_LINE 64

class Matrix {

private:

public:

    int rows{}, columns{};

    float *data{};

    Matrix(int rows, int columns);

    ~Matrix();

    Matrix* transposed();
    
    Matrix* normalized();
    
    Matrix* sumRows();
    
    Matrix* variance0Axis();

    Matrix* mean0Axis();

    Matrix *multiply(Matrix *W);

    Matrix *sum(Matrix *W, float scalar);

    Matrix *elemMul(Matrix *W);

    Matrix *elemMulVector(Matrix *W, Matrix *W1);

    Matrix *elemMulVector(Matrix *W);

    Matrix *centralized(Matrix *desiredMean);

    Matrix *copy();

    float sumElements();

    void randomize();

    static Matrix *invDeviation(Matrix *desiredVar);

    void set(Matrix *W);

    Matrix *ReLUDerivative(Matrix *W);

    Matrix *iam2cool(int filterSize, int stride);

    Matrix *cool2ami(int filters, int batchSize);
};


#endif //NNCPP_EXEC_MATRIX_H
