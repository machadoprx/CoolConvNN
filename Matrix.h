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

#define CACHE_LINE 32

class Matrix {

private:

public:
    int rows{}, columns{};
    float *data{};

    Matrix(int rows, int columns);
    ~Matrix();
    
    Matrix *transposed();
    Matrix *normalized();
    Matrix *normalized2(Matrix *mean, Matrix *deviationInv);
    Matrix *sumRows();
    Matrix* sumColumns();
    Matrix *variance0Axis(Matrix *mean);
    Matrix *mean0Axis();
    Matrix *multiply(Matrix *W, bool transA, bool transB);
    Matrix *sum(Matrix *W, float scalar);
    Matrix *elemMul(Matrix *W);
    Matrix *elemMulVector(Matrix *W, Matrix *W1);
    Matrix *elemMulVector(Matrix *W);
    Matrix *ReLUDerivative(Matrix *W);
    Matrix *copy();
    
    float sumElements();
    
    void normalize2(Matrix *mean, Matrix *deviationInv);
    void normalize();
    void randomize(float mean, float deviation);
    void set(Matrix *W);
    void setArray(float *data);
    void setRow(float *row, int rowPos);
    void accumulate(Matrix *W);
    void apply_sum(Matrix *W, float scalar);
    void apply_relu();
    void apply_reluderivative(Matrix* W);

    static Matrix *invDeviation(Matrix *variance);
    static float ReLU(float x);
    static void mcopy(float *dest, float *src, int size);
};

#endif //NNCPP_EXEC_MATRIX_H
