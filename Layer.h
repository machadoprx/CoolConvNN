//
// Created by vmachado on 2/11/20.
//

#ifndef NNCPP_EXEC_LAYER_H
#define NNCPP_EXEC_LAYER_H

#include "Matrix.h"

class Layer {

private:
    void validationInput(Matrix* &rawInput);
    void trainingInput(Matrix* &rawInput);
    void updateRunningStatus(Matrix *mean, Matrix *variance);
    Matrix *getBatchNormDerivative(Matrix *dOut);
    void updateWeights(Matrix *dWeights, float learningRate);
    void updateGammaBeta(Matrix *dGamma, Matrix *dBeta, float learningRate);

public:
    Matrix *weights{}, *gamma{}, *beta{};
    Matrix *runningMean{}, *runningVariance{}, *deviationInv{};
    Matrix *input{}, *inputNormalized{};
    bool hidden, isFirst;
    Layer(int inputDimension, int outputDimension, bool hidden, bool isFirst);
    Layer(int inputDimension, int outputDimension, bool hidden, bool isFirst, float *weights, float *gamma, float *beta,
          float *runningMean, float *runningVariance);
    ~Layer();
    Matrix *getWeights();
    Matrix *getDeviationInv();
    Matrix *getGamma();
    Matrix *getBeta();
    Matrix *getRunningMean();
    Matrix *getRunningVariance();
    Matrix *feedForward(Matrix *rawInput, bool validation);
    Matrix *backPropagation(Matrix *dOut, float lambdaReg, float learningRate);
};


#endif //NNCPP_EXEC_LAYER_H
