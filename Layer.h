//
// Created by vmachado on 2/11/20.
//

#ifndef NNCPP_EXEC_LAYER_H
#define NNCPP_EXEC_LAYER_H

#include "Matrix.h"

class Layer {

private:

    Matrix *deviationInv{};

    bool frozen = false;

    void validationOutput();

    void trainingOutput();

    static double ReLU(double);

public:

    Matrix *weights{}, *output{}, *outputNormalized{}, *gamma{}, *beta{};

    Matrix *runningMean{}, *runningVariance{};

    int outputDimension{}, inputDimension{};

    Layer(int, int);

    Layer(int inputDimension, int outputDimension, double *weights, double *gamma, double *beta,
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

    void updateRunningStatus(Matrix *mean, Matrix *variance);

    void feedForward(Matrix *input, bool hidden, bool validation);

    void updateWeights(Matrix *dWeights, double learningRate);

    void updateGammaBeta(Matrix *dGamma, Matrix *dBeta, double learningRate);
};


#endif //NNCPP_EXEC_LAYER_H
