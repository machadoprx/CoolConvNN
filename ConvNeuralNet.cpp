//
// Created by vmachado on 2/11/20.
//

#include "ConvNeuralNet.h"

ConvNeuralNet::ConvNeuralNet(const char* cnnFileName) {

    FILE *cnnFile = fopen(cnnFileName, "r");
    fscanf(cnnFile, "%d\n", &convPoolLayers);
    
    for (int i = 0; i < convPoolLayers; i++) {
        int inputChannels, outputChannels, stride, filterSize, padding, inputWidth, inputHeight;
        fscanf(cnnFile, "%d %d %d %d %d %d %d\n", &inputChannels, &outputChannels, &stride, &filterSize, &padding, &inputWidth, &inputHeight);
        convLayers.push_back(new ConvLayer(inputChannels, outputChannels, stride, filterSize, padding, inputWidth, inputHeight));
        fscanf(cnnFile, "%d %d %d %d %d %d\n", &stride, &filterSize, &inputChannels, &padding, &inputWidth, &inputHeight);
        poolLayers.push_back(new PoolLayer(stride, filterSize, inputChannels, padding, inputWidth, inputHeight));
    }

    fscanf(cnnFile, "%d\n", &fcAdditionalHidden);
    fscanf(cnnFile, "%d %d %d\n", &fcFeaturesDim, &fcOutputDim, &fcLayersDim);

    fcLayers.push_back(new Layer(fcFeaturesDim, fcLayersDim, true, false));
    for (int i = 0; i < fcAdditionalHidden; i++) {
        fcLayers.push_back(new Layer(fcLayersDim, fcLayersDim, true, false));
    }
    fcLayers.push_back(new Layer(fcLayersDim, fcOutputDim, false, false));

    fscanf(cnnFile, "%d %f\n", &batchSize, &learningRate);
    fclose(cnnFile);
}

ConvNeuralNet::ConvNeuralNet(const char* cnnFileName, const char* weightsFileName) { // array alignment

    FILE *cnnFile = fopen(cnnFileName, "r");
    FILE *weightsFile = fopen(weightsFileName, "rb");

    fscanf(cnnFile, "%d\n", &convPoolLayers);
    float *weights, *beta, *gamma, *runningMean, *runningVariance;
    
    for (int i = 0; i < convPoolLayers; i++) {
        int inputChannels, outputChannels, stride, filterSize, padding, inputWidth, inputHeight;
        fscanf(cnnFile, "%d %d %d %d %d %d %d\n", &inputChannels, &outputChannels, &stride, &filterSize, &padding, &inputWidth, &inputHeight);
        
        convLayers.push_back(new ConvLayer(inputChannels, outputChannels, stride, filterSize, padding, inputWidth, inputHeight));
        
        weights = new float[outputChannels * filterSize * filterSize * inputChannels];
        beta = new float[outputChannels];

        fread(weights, sizeof(float) * outputChannels * filterSize * filterSize * inputChannels, 1, weightsFile);
        fread(beta, sizeof(float) * outputChannels, 1, weightsFile);

        convLayers.at(i)->filters->setArray(weights);
        convLayers.at(i)->bias->setArray(beta);

        delete weights;
        delete beta;

        fscanf(cnnFile, "%d %d %d %d %d %d\n", &stride, &filterSize, &inputChannels, &padding, &inputWidth, &inputHeight);
        poolLayers.push_back(new PoolLayer(stride, filterSize, inputChannels, padding, inputWidth, inputHeight));
    }

    fscanf(cnnFile, "%d\n", &fcAdditionalHidden);
    fscanf(cnnFile, "%d %d %d\n", &fcFeaturesDim, &fcOutputDim, &fcLayersDim);

    weights = new float[fcFeaturesDim * fcLayersDim];
    gamma = new float[fcFeaturesDim];
    beta = new float[fcFeaturesDim];
    runningMean = new float[fcFeaturesDim];
    runningVariance = new float[fcFeaturesDim];

    fread(weights, sizeof(float) * fcFeaturesDim * fcLayersDim, 1, weightsFile);
    fread(gamma, sizeof(float) * fcFeaturesDim, 1, weightsFile);
    fread(beta, sizeof(float) * fcFeaturesDim, 1, weightsFile);
    fread(runningMean, sizeof(float) * fcFeaturesDim, 1, weightsFile);
    fread(runningVariance, sizeof(float) * fcFeaturesDim, 1, weightsFile);

    fcLayers.push_back(new Layer(fcFeaturesDim, fcLayersDim, true, false, weights, gamma, beta, runningMean, runningVariance));

    delete weights;
    delete gamma;
    delete beta;
    delete runningVariance;
    delete runningMean;

    for (int i = 0; i < fcAdditionalHidden; i++) {
        weights = new float[fcLayersDim * fcLayersDim];
        gamma = new float[fcLayersDim];
        beta = new float[fcLayersDim];
        runningMean = new float[fcLayersDim];
        runningVariance = new float[fcLayersDim];

        fread(weights, sizeof(float) * fcLayersDim * fcLayersDim, 1, weightsFile);
        fread(gamma, sizeof(float) * fcLayersDim, 1, weightsFile);
        fread(beta, sizeof(float) * fcLayersDim, 1, weightsFile);
        fread(runningMean, sizeof(float) * fcLayersDim, 1, weightsFile);
        fread(runningVariance, sizeof(float) * fcLayersDim, 1, weightsFile);

        fcLayers.push_back(new Layer(fcLayersDim, fcLayersDim, true, false, weights, gamma, beta, runningMean, runningVariance));

        delete weights;
        delete gamma;
        delete beta;
        delete runningVariance;
        delete runningMean;
    }
    weights = new float[fcLayersDim * fcOutputDim];
    gamma = new float[fcLayersDim];
    beta = new float[fcLayersDim];
    runningMean = new float[fcLayersDim];
    runningVariance = new float[fcLayersDim];

    fread(weights, sizeof(float) * fcOutputDim * fcLayersDim, 1, weightsFile);
    fread(gamma, sizeof(float) * fcLayersDim, 1, weightsFile);
    fread(beta, sizeof(float) * fcLayersDim, 1, weightsFile);
    fread(runningMean, sizeof(float) * fcLayersDim, 1, weightsFile);
    fread(runningVariance, sizeof(float) * fcLayersDim, 1, weightsFile);

    fcLayers.push_back(new Layer(fcLayersDim, fcOutputDim, false, false, weights, gamma, beta, runningMean, runningVariance));

    delete weights;
    delete gamma;
    delete beta;
    delete runningVariance;
    delete runningMean;

    fscanf(cnnFile, "%d %f\n", &batchSize, &learningRate);
    fclose(cnnFile);
    fclose(weightsFile);
}

ConvNeuralNet::~ConvNeuralNet() {
    for (auto & layer : fcLayers) {
        delete layer;
    }
    for (auto & layer : convLayers) {
        delete layer;
    }
    for (auto & layer : poolLayers) {
        delete layer;
    }
}

void ConvNeuralNet::saveState(const char* weightsFileName) {

    FILE *weightsFile = fopen(weightsFileName, "wb");

    for (int i = 0; i < convPoolLayers; i++) {
        auto layer = convLayers.at(i);

        fwrite(layer->filters->data, sizeof(float) * layer->outputChannels * layer->filterSize * layer->filterSize * layer->inputChannels, 1, weightsFile);
        fwrite(layer->bias->data, sizeof(float) * layer->outputChannels, 1, weightsFile);
    }

    Layer *layer = fcLayers.at(0);
    fwrite(layer->weights->data, sizeof(float) * fcFeaturesDim * fcLayersDim, 1, weightsFile);
    fwrite(layer->gamma->data, sizeof(float) * fcFeaturesDim, 1, weightsFile);
    fwrite(layer->beta->data, sizeof(float) * fcFeaturesDim, 1, weightsFile);
    fwrite(layer->runningMean->data, sizeof(float) * fcFeaturesDim, 1, weightsFile);
    fwrite(layer->runningVariance->data, sizeof(float) * fcFeaturesDim, 1, weightsFile);

    int i = 1;
    for (; i < (int) fcLayers.size() - 1; i++) {
        layer = fcLayers.at(i);
        fwrite(layer->weights->data, sizeof(float) * fcLayersDim * fcLayersDim, 1, weightsFile);
        fwrite(layer->gamma->data, sizeof(float) * fcLayersDim, 1, weightsFile);
        fwrite(layer->beta->data, sizeof(float) * fcLayersDim, 1, weightsFile);
        fwrite(layer->runningMean->data, sizeof(float) * fcLayersDim, 1, weightsFile);
        fwrite(layer->runningVariance->data, sizeof(float) * fcLayersDim, 1, weightsFile);
    }

    layer = fcLayers.at(i);
    fwrite(layer->weights->data, sizeof(float) * fcOutputDim * fcLayersDim, 1, weightsFile);
    fwrite(layer->gamma->data, sizeof(float) * fcLayersDim, 1, weightsFile);
    fwrite(layer->beta->data, sizeof(float) * fcLayersDim, 1, weightsFile);
    fwrite(layer->runningMean->data, sizeof(float) * fcLayersDim, 1, weightsFile);
    fwrite(layer->runningVariance->data, sizeof(float) * fcLayersDim, 1, weightsFile);

    fclose(weightsFile);

}

Matrix* ConvNeuralNet::forwardStep(Matrix* batch, bool validation) {

    int i;
    Matrix *tmp = convLayers.at(0)->feedForward(batch);
    Matrix *curr = poolLayers.at(0)->feedForward(tmp);
    delete tmp;
    
    for (i = 1; i < (int)convLayers.size(); i++) {
        tmp = convLayers.at(i)->feedForward(curr);
        delete curr;
        curr = poolLayers.at(i)->feedForward(tmp);
        delete tmp;
    }

    for (i = 0; i < (int)fcLayers.size(); i++) {
        tmp = fcLayers.at(i)->feedForward(curr, validation);
        delete curr;
        curr = tmp;
    }

    auto prob = curr->normalized();
    delete curr;

    return prob;
}

void ConvNeuralNet::backPropagationStep(Matrix* prob, Matrix* batch, int *labels) {
    
    int i;
    Matrix *tmp;
    auto dOut = NeuralNet::getProbDerivative(prob, labels);

    for (i = fcLayers.size() - 1; i >= 0; --i) {
        tmp = fcLayers.at(i)->backPropagation(dOut, lambdaReg, learningRate);
        delete dOut;
        dOut = tmp;
    }

    for (i = convLayers.size() - 1; i >= 0; --i) {
        tmp = poolLayers.at(i)->backPropagation(dOut);
        delete dOut;
        dOut = convLayers.at(i)->backPropagation(tmp, learningRate);
        delete tmp;
    }

    delete dOut;
}

void ConvNeuralNet::train(float** &dataSet, int* &labels, int samples, int epochs){

    assert(samples >= batchSize);

    int numberOfBatches = samples % batchSize != 0 ?
                          (samples / batchSize) + 1
                          : samples / batchSize;

    for (int e = 1; e <= epochs; e++) {

        int dataIndex = 0;
        float loss = 0;

        // shuffle dataset
        NeuralNet::shuffleDataFisherYates(dataSet, labels, samples);

        for (int k = 0; k < numberOfBatches; k++) {

            // prepare batch
            int batchLength;
            if (dataIndex + batchSize >= samples) {
                batchLength = samples - dataIndex;
            }
            else {
                batchLength = batchSize;
            }

            ConvLayer *in = convLayers.at(0);
            int inputDim = in->inputChannels * in->inputWidth * in->inputHeight;
            auto batch = new Matrix(batchLength, inputDim);
            auto batchLabels = new int[batchLength];

            NeuralNet::prepareBatch(dataSet, labels, batchLength, dataIndex, batch, batchLabels, inputDim);

            //forward step
            auto score = forwardStep(batch, false);

            // get correct probabilities for each class
            auto correctProb = NeuralNet::getCorrectProb(score, batchLabels);

            // compute loss
            loss += NeuralNet::getDataLoss(correctProb) + NeuralNet::getRegulationLoss(fcLayers, lambdaReg);

            // backpropagation step
            backPropagationStep(score, batch, batchLabels);

            // update data index
            dataIndex += batchLength;

            //clean
            delete batch;
            delete[] batchLabels;
            delete score;
            delete correctProb;
        }
        std::cout << "epoch: " << e << " loss: " << loss / numberOfBatches << '\n';
    }
}