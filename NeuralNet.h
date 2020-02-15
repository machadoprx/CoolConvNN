//
// Created by vmachado on 2/11/20.
//

#ifndef NNCPP_EXEC_NEURALNET_H
#define NNCPP_EXEC_NEURALNET_H

#include "Layer.h"
#include <ctime>
#include <fstream>
#include <iostream>
#include <cstdlib>

class NeuralNet {

public:
    NeuralNet(int featuresDimension, int outputDimension, int additionalHiddenLayers, int layersDimension,
              int batchSize, double learningRate);

    explicit NeuralNet(const char *path);

    ~NeuralNet();

    void saveState(const char *path);

    Matrix *forwardStep(Matrix *batch, bool validation);

    void train(double** &dataSet, int* &labels, int samples, int epochs);

    void setLearningRate(double value);

private:

    int featuresDimension{};
    int outputDimension{};
    int additionalHiddenLayers{};
    int layersDimension{};
    int batchSize{};
    double lambdaReg = 1e-3;
    double learningRate = 0.01;
    std::vector<Layer*> layers;

    double getRegulationLoss();

    static void shuffleDataFisherYates(double** &data, int* labels, int samples);

    static Matrix *getCorrectProb(Matrix *prob, int const *labels);

    static Matrix *getProbDerivative(Matrix *prob, int const *labels);

    void backPropagationStep(Matrix *prob, Matrix *batch, int const *labels);

    static double getDataLoss(Matrix *correctProb);
};


#endif //NNCPP_EXEC_NEURALNET_H
