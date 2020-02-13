//
// Created by vmachado on 2/11/20.
//

#ifndef NNCPP_EXEC_NEURALNET_H
#define NNCPP_EXEC_NEURALNET_H

#include "Layer.h"
#include <ctime>
#include <fstream>
#include <iostream>

class NeuralNet {

public:
    NeuralNet(int featuresDimension, int outputDimension, int additionalHiddenLayers, int layersDimension,
              int batchSize);

    explicit NeuralNet(const char *path);

    ~NeuralNet();

    void saveState(const char *path);

    Matrix *forwardStep(Matrix *batch, bool validation);

    void train(double** &dataSet, int* &labels, int samples, int epochs);

private:
    std::vector<Layer*> layers;
    double lambdaReg = 1e-3;
    double learningRate = 0.1;
    int featuresDimension{};
    int outputDimension{};
    int additionalHiddenLayers{};
    int layersDimension{};
    int batchSize{};

    double getRegulationLoss();

    Matrix *getCorrectProb(Matrix *prob, int *labels);

    Matrix *getProbDerivative(Matrix *prob, int *labels);

    Matrix *getReLUDerivative(Matrix *W, Matrix *W1);

    double getDataLoss(Matrix *correctProb);

    void shuffleDataFisherYates(double** &data, int* &labels, int samples);

    Matrix *getBatchNormDerivative(Matrix *dOut, Layer *layer);

    void backPropagationStep(Matrix *prob, Matrix *batch, int *labels);


};


#endif //NNCPP_EXEC_NEURALNET_H
