//
// Created by vmachado on 2/11/20.
//

#ifndef NNCPP_EXEC_LAYER_H
#define NNCPP_EXEC_LAYER_H

#include "Matrix.h"

class Layer {

private:

    bool hidden;

    bool frozen = false;

    void validationOutput();

    void trainingOutput();

    double ReLU(double);

    void updateRunningStatus(Matrix *mean, Matrix *variance);

    static Matrix *getBatchNormDerivative(Matrix *dOut, Layer* prev);

public:

    Matrix *weights{}, *output{}, *outputNormalized{}, *gamma{}, *beta{};

    Matrix *runningMean{}, *runningVariance{}, *deviationInv{};

    Layer(int inputDimension, int outputDimension, bool hidden);

    Layer(int inputDimension, int outputDimension, bool hidden, double *weights, double *gamma, double *beta,
          double *runningMean, double *runningVariance);

    ~Layer();

    Matrix* getOutput();

    Matrix *getWeights();

    Matrix *getDeviationInv();

    Matrix *getOutputNormalized();

    Matrix *getGamma();

    Matrix *getBeta();

    Matrix *getRunningMean();

    Matrix *getRunningVariance();

    void setFrozen(bool frozen);

    void feedForward(Matrix *input, bool validation);

    void updateWeights(Matrix *dWeights, double learningRate);

    void updateGammaBeta(Matrix *dGamma, Matrix *dBeta, double learningRate);

    Matrix *backPropagation(Matrix *dOut, Matrix *input, Layer* previous, double learningRate, double lambdaReg);

};


#endif //NNCPP_EXEC_LAYER_H
