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

    void updateRunningStatus(Matrix*, Matrix*);

public:

    Matrix *weights{}, *output{}, *outputNormalized{}, *gamma{}, *beta{};

    Matrix *runningMean{}, *runningVariance{};

    int outputDimension{}, inputDimension{};

    Layer(int, int);

    Layer(int, int, double*, double*, double*, double*, double*);

    ~Layer();

    void feedForward(Matrix*, bool, bool);

    void updateWeights(Matrix*, double);

    void updateGammaBeta(Matrix*, Matrix*, double);

    Matrix* getOutput();

    Matrix *getWeights();

    Matrix *getDeviationInv();

    Matrix *getOutputNormalized();

    Matrix *getGamma();

    Matrix *getBeta();

    Matrix *getRunningMean();

    Matrix *getRunningVariance();

    void setFrozen(bool frozen);
};


#endif //NNCPP_EXEC_LAYER_H
