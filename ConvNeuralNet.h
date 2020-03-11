#ifndef NNCPP_EXEC_CONVNEURALNET_H
#define NNCPP_EXEC_CONVNEURALNET_H

#include "Layer.h"
#include "ConvLayer.h"
#include "NeuralNet.h"
#include "PoolLayer.h"
#include <ctime>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>

class ConvNeuralNet {

public:
    ConvNeuralNet(const char* cnnFileName);
    ConvNeuralNet(const char* cnnFileName, const char* weightsFileName);
    ~ConvNeuralNet();

    Matrix *forwardStep(Matrix *batch, bool validation);
    void saveState(const char* weightsFileName);
    void train(float** &dataSet, int* &labels, int samples, int epochs);
    
private:
    int fcFeaturesDim{}, fcOutputDim, fcLayersDim, fcAdditionalHidden, convPoolLayers, batchSize;
    float lambdaReg = 1e-3;
    float learningRate = 0.01;
    
    std::vector<Layer*> fcLayers;
    std::vector<ConvLayer*> convLayers;
    std::vector<PoolLayer*> poolLayers;

    float getRegulationLoss();
    void backPropagationStep(Matrix *prob, Matrix *batch, int *labels);
};


#endif //NNCPP_EXEC_NEURALNET_H
