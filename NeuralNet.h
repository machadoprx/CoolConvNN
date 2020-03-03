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
              int batchSize, float learningRate);
    explicit NeuralNet(const char *path);
    NeuralNet();
    ~NeuralNet();
    void saveState(const char *path);
    Matrix *forwardStep(Matrix *batch, bool validation);
    void train(float** &dataSet, int* &labels, int samples, int epochs);
    void setLearningRate(float value);
private:
    int featuresDimension{};
    int outputDimension{};
    int additionalHiddenLayers{};
    int layersDimension{};
    int batchSize{};
    float lambdaReg = 1e-3;
    float learningRate = 0.01;
    std::vector<Layer*> layers;
    float getRegulationLoss();
    static void shuffleDataFisherYates(float** &data, int* labels, int samples);
    static Matrix *getCorrectProb(Matrix *prob, int *labels);
    static Matrix *getProbDerivative(Matrix *prob, int *labels);
    void backPropagationStep(Matrix *prob, Matrix *batch, int *labels);
    void prepareBatch(float** &dataSet, int* &labels, int batchLength, int dataIndex, Matrix *batch, int *batchLabels);
    static float getDataLoss(Matrix *correctProb);
};


#endif //NNCPP_EXEC_NEURALNET_H
