//
// Created by vmachado on 2/11/20.
//

#ifndef NNCPP_EXEC_LAYER_H
#define NNCPP_EXEC_LAYER_H

#include "Matrix.h"

class Layer {

private:

    bool hidden;

    void validationOutput();

    void trainingOutput();

    float ReLU(float);

    void updateRunningStatus(Matrix *mean, Matrix *variance);

    static Matrix *getBatchNormDerivative(Matrix *dOut, Layer* prev);

public:

    Matrix *weights{}, *output{}, *outputNormalized{}, *gamma{}, *beta{};

    Matrix *runningMean{}, *runningVariance{}, *deviationInv{};

    Layer(int inputDimension, int outputDimension, bool hidden);

    Layer(int inputDimension, int outputDimension, bool hidden, float *weights, float *gamma, float *beta,
          float *runningMean, float *runningVariance);

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

    void updateWeights(Matrix *dWeights, float learningRate);

    void updateGammaBeta(Matrix *dGamma, Matrix *dBeta, float learningRate);

    Matrix *backPropagation(Matrix *dOut, Matrix* &dWeights, Matrix* &dGamma, Matrix* &dBeta, Matrix *input, Layer* previous, float lambdaReg);

};


#endif //NNCPP_EXEC_LAYER_H
