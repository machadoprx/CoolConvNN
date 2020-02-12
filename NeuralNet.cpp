//
// Created by vmachado on 2/11/20.
//

#include "NeuralNet.h"

NeuralNet::NeuralNet(int featuresDimension,
                    int outputDimension,
                    int additionalHiddenLayers,
                    int layersDimension,
                    int batchSize){

    this->featuresDimension      = featuresDimension;
    this->outputDimension        = outputDimension;
    this->additionalHiddenLayers = additionalHiddenLayers;
    this->layersDimension        = layersDimension;
    this->batchSize              = batchSize;

    auto *inputLayer = new Layer(featuresDimension, layersDimension); // (n, d) @ (d , h)
    layers.push_back(inputLayer);
    for (int i = 0; i < additionalHiddenLayers; i++) {
        auto *hiddenLayer = new Layer(layersDimension, layersDimension); // (n, h) @ (h , h)
        layers.push_back(hiddenLayer);
    }
    auto *outputLayer = new Layer(layersDimension, outputDimension); // (n, h) @ (h , k)
    layers.push_back(outputLayer);
}

NeuralNet::NeuralNet(const char *path) {

    FILE *f = fopen(path, "rb");

    if (f == nullptr) {
        throw std::runtime_error("invalid file");
    }

    fread(&additionalHiddenLayers, sizeof(int), 1, f);
    fread(&featuresDimension, sizeof(int), 1, f);
    fread(&layersDimension, sizeof(int), 1, f);
    fread(&outputDimension, sizeof(int), 1, f);

    auto *inputLayer = new Layer(featuresDimension, layersDimension);
    fread(inputLayer->weights, sizeof(double) * featuresDimension * layersDimension, 1, f);
    fread(inputLayer->gamma, sizeof(double) * layersDimension, 1, f);
    fread(inputLayer->beta, sizeof(double) * layersDimension, 1, f);
    fread(inputLayer->runningMean, sizeof(double) * layersDimension, 1, f);
    fread(inputLayer->runningVariance, sizeof(double) * layersDimension, 1, f);
    layers.push_back(inputLayer);

    for (int i = 0; i < additionalHiddenLayers; i++) {

        auto *hiddenLayer = new Layer(layersDimension, layersDimension);
        fread(hiddenLayer->weights, sizeof(double) * layersDimension * layersDimension, 1, f);
        fread(hiddenLayer->gamma, sizeof(double) * layersDimension, 1, f);
        fread(hiddenLayer->beta, sizeof(double) * layersDimension, 1, f);
        fread(hiddenLayer->runningMean, sizeof(double) * layersDimension, 1, f);
        fread(hiddenLayer->runningVariance, sizeof(double) * layersDimension, 1, f);
        layers.push_back(hiddenLayer);

    }

    auto *outputLayer = new Layer(layersDimension, outputDimension);
    fread(outputLayer->weights, sizeof(double) * layersDimension * outputDimension, 1, f);
    fread(outputLayer->gamma, sizeof(double) * outputDimension, 1, f);
    fread(outputLayer->beta, sizeof(double) * outputDimension, 1, f);
    fread(outputLayer->runningMean, sizeof(double) * outputDimension, 1, f);
    fread(outputLayer->runningVariance, sizeof(double) * outputDimension, 1, f);
    layers.push_back(outputLayer);

}

void NeuralNet::saveState(const char *path) {

    FILE *f = fopen(path, "wb");

    if (f == nullptr) {
        throw std::runtime_error("invalid file");
    }

    fwrite(&additionalHiddenLayers, sizeof(int), 1, f);
    fwrite(&featuresDimension, sizeof(int), 1, f);
    fwrite(&layersDimension, sizeof(int), 1, f);
    fwrite(&outputDimension, sizeof(int), 1, f);

    auto *inputLayer = layers.at(0);
    fwrite(inputLayer->weights, sizeof(double) * featuresDimension * layersDimension, 1, f);
    fwrite(inputLayer->gamma, sizeof(double) * layersDimension, 1, f);
    fwrite(inputLayer->beta, sizeof(double) * layersDimension, 1, f);
    fwrite(inputLayer->runningMean, sizeof(double) * layersDimension, 1, f);
    fwrite(inputLayer->runningVariance, sizeof(double) * layersDimension, 1, f);

    int i = 1;
    for (; i < layers.size() - 1; i++) {
        auto *hiddenLayer = layers.at(i);
        fwrite(hiddenLayer->weights, sizeof(double) * layersDimension * layersDimension, 1, f);
        fwrite(hiddenLayer->gamma, sizeof(double) * layersDimension, 1, f);
        fwrite(hiddenLayer->beta, sizeof(double) * layersDimension, 1, f);
        fwrite(hiddenLayer->runningMean, sizeof(double) * layersDimension, 1, f);
        fwrite(hiddenLayer->runningVariance, sizeof(double) * layersDimension, 1, f);
    }

    auto *outputLayer = layers.at(i);
    fwrite(outputLayer->weights, sizeof(double) * layersDimension * outputDimension, 1, f);
    fwrite(outputLayer->gamma, sizeof(double) * outputDimension, 1, f);
    fwrite(outputLayer->beta, sizeof(double) * outputDimension, 1, f);
    fwrite(outputLayer->runningMean, sizeof(double) * outputDimension, 1, f);
    fwrite(outputLayer->runningVariance, sizeof(double) * outputDimension, 1, f);

}

Matrix* NeuralNet::getCorrectProb(Matrix *prob, const int *labels){

    auto *correctProb = new Matrix(prob->rows, 1);

    for (int i = 0; i < prob->rows; i++) {
        correctProb->data[i] = (-1) * log(prob->data[i * prob->rows + labels[i]]);
    }

    return correctProb;
}

Matrix* NeuralNet::getProbDerivative(Matrix *prob, const int *batchLabels){

    auto *dProb = new Matrix(prob->rows, prob->columns);

    for (int i = 0; i < prob->rows; i++) {

        dProb->data[i * dProb->columns + batchLabels[i]] -= 1;

        for (int j = 0; j < prob->columns; j++) {

            int index = i * prob->columns + j;

            dProb->data[index] = (prob->data[index] + dProb->data[index]) / prob->rows;
        }
    }

    return dProb;
}

Matrix* NeuralNet::getReLUDerivative(Matrix *W, Matrix *W1) {

    auto *R = new Matrix(W->rows, W->columns);

    for (int i = 0; i < R->rows; i++) {

        for (int j = 0; j < R->columns; j++) {

            int index = i * W1->columns + j;
            R->data[index] = W1->data[index] <= 0 ? 0 : W->data[index];
        }
    }

    return R;
}

double NeuralNet::getDataLoss(Matrix *correctProb, int n){

    double loss = 0;

    for (int i = 0; i < correctProb->rows; i++) {
        loss += correctProb->data[i];
    }
    loss = loss / n;

    return loss;
}

double NeuralNet::getRegulationLoss(){

    double regLoss = 0;

    for (auto & layer : layers) {
        Matrix *weights = layer->getWeights();
        Matrix *w2 = weights->hadamard(weights);
        double t = w2->sumElements();
        regLoss += 0.5 * lambdaReg * t;
        delete weights;
        delete w2;
    }

    return regLoss;
}

void NeuralNet::shuffleDataFisherYates(double **data, int *labels, int samples) {

    srand(time(NULL));
    auto *tmpData = new double[featuresDimension];
    int tmpLabel, randomIndex;

    for (int i = samples - 1; i >= 1; i--) {

        randomIndex = rand() % (i + 1);
        memcpy(tmpData, data[i], featuresDimension);
        tmpLabel = labels[i];

        memcpy(data[i], data[randomIndex], featuresDimension);
        labels[i] = labels[randomIndex];

        memcpy(data[randomIndex], tmpData, featuresDimension);
        labels[randomIndex] = tmpLabel;
    }

    delete[] tmpData;
}

Matrix* NeuralNet::getBatchNormDerivative(Matrix *dOut, Layer *layer) {

    // https://kevinzakka.github.io/2016/09/14/batch_normalization/

    int N = dOut->rows;
    auto *oNormalized = layer->getOutputNormalized();
    auto *deviationInv = layer->getDeviationInv();
    auto *gamma = layer->getGamma();

    auto *dBatch0 = new Matrix(dOut->rows, dOut->columns);
    auto *dBatch1 = new Matrix(dOut->rows, dOut->columns);
    auto *dBatch2 = new double[dOut->columns];
    auto *dBatch3 = new double[dOut->columns];

    for (int i = 0; i < dOut->rows; i++) {
        for (int j = 0; j < dOut->columns; j++) {
            int index = i * dOut->columns + j;
            double dONormRowElem = dOut->data[index] * gamma->data[j];
            dBatch1->data[index] = dONormRowElem * N;
            dBatch2[j] += dONormRowElem;
            dBatch3[j] += dONormRowElem * oNormalized->data[index];
        }
    }

    for (int i = 0; i < dOut->rows; i++) {
        for (int j = 0; j < dOut->columns; j++) {
            int index = i * dOut->columns + j;
            dBatch1->data[index] = dBatch1->data[index] - dBatch2[j];
            dBatch1->data[index] = dBatch1->data[index] - (oNormalized->data[index] * dBatch3[j]);
            dBatch0->data[index] = dBatch1->data[index] * deviationInv->data[j] * (1.0 / (double) N);
        }
    }

    delete oNormalized;
    delete deviationInv;
    delete gamma;
    delete dBatch1;
    delete[] dBatch2;
    delete[] dBatch3;
    return dBatch0;
}

Matrix* NeuralNet::forwardStep(Matrix *batch, bool validation) {

    size_t layersSize = layers.size() - 1;
    Layer *current = layers.at(0), *previous;
    current->feedForward(batch, true, validation);

    for (int i = 1; i < layersSize; i++) {
        previous = current;
        current = layers.at(i);
        Matrix *prevO = previous->getOutput();
        current->feedForward(prevO, true, validation);
        delete prevO;
    }

    previous = current;
    Matrix *prevO = previous->getOutput();
    current = layers.at(layersSize);
    current->feedForward(prevO, false, validation);
    delete prevO;

    Matrix *out = current->getOutput();
    Matrix *prob = out->normalized();
    delete out;

    return prob;
}

void NeuralNet::backPropagationStep(Matrix *prob, Matrix *batch, int *labels) {

    size_t layersSize = layers.size() - 1;
    Matrix *dCurrO = getProbDerivative(prob, labels);
    Layer *current, *previous;

    for (int i = layersSize; i >= 1; i--) {

        current = layers.at(i);
        previous = layers.at(i - 1);

        Matrix *prevO = previous->getOutput();
        Matrix *currW = current->getWeights();
        Matrix *currWT = currW->transposed();
        Matrix *prevOT = prevO->transposed();

        // Current derivative with L2 regularization
        Matrix *dCurrW = prevOT->multiply(dCurrO);
        Matrix *dCurrWReg = dCurrW->sum(currW, lambdaReg);

        // update current layer weights
        current->updateWeights(dCurrWReg, learningRate);

        // gamma and beta derivative for batch norm
        Matrix *dPrevO = dCurrO->multiply(currWT);
        Matrix *prevONorm = previous->getOutputNormalized();
        Matrix *dGammaPartial = dPrevO->hadamard(prevONorm);
        Matrix *dGamma = dGammaPartial->sumRows();
        Matrix *dBeta = dPrevO->sumRows();

        // update previous layer batch norm weights
        previous->updateGammaBeta(dGamma, dBeta, learningRate);

        // get prev layer output final derivative
        Matrix *dPrevONorm = getBatchNormDerivative(dPrevO, previous);

        delete dCurrO;
        dCurrO = getReLUDerivative(dPrevONorm, prevO);

        //clear
        delete currW;
        delete prevO;
        delete currWT;
        delete prevOT;
        delete dCurrW;
        delete dCurrWReg;
        delete dPrevO;
        delete dGammaPartial;
        delete dGamma;
        delete dBeta;
        delete dPrevONorm;
        delete prevONorm;
    }

    current = layers.at(0);

    Matrix *currW = current->getWeights();
    Matrix *dataOT = batch->transposed();
    Matrix *dCurrW = dataOT->multiply(dCurrO);
    Matrix *dCurrWReg = dCurrW->sum(currW, lambdaReg);

    current->updateWeights(dCurrWReg, learningRate);

    // clear
    delete dCurrO;
    delete currW;
    delete dataOT;
    delete dCurrW;
    delete dCurrWReg;
}

void NeuralNet::train(double **dataSet, int *labels, int samples, int epochs){

    assert(samples >= batchSize);

    int numberOfBatches = samples % batchSize != 0 ?
                          (samples / batchSize) + 1
                          : samples / batchSize;

    for (int e = 0; e < epochs; e++) {

        int dataIndex = 0;
        double loss = 0;
        // shuffle dataset
        shuffleDataFisherYates(dataSet, labels, samples);

        for (int k = 0; k < numberOfBatches; k++) {

            // prepare batch
            Matrix *batch;
            int *batchLabels;
            int batchLength = batchSize;

            if (dataIndex + batchSize >= samples){
                batchLength = samples - dataIndex;
            }

            batch = new Matrix(batchLength, featuresDimension);
            batchLabels = new int[batchLength];
            memset(batchLabels, 0, sizeof(int) * batchLength);

            double *posPtr = batch->data;
            for (int i = dataIndex; i < dataIndex + batchLength; i++) {
                memcpy(posPtr, dataSet[i], featuresDimension);
                posPtr += featuresDimension;
            }
            memcpy(batchLabels, labels, batchLength);

            //forward step
            Matrix *score = forwardStep(batch, false);

            // get correct probabilities for each class
            Matrix *correctProb = getCorrectProb(score, batchLabels);

            // compute loss
            // L = 1/n * sum (loss) + for each layer(1/2 * lambda * sum (layer weight @ layer weight))
            loss += getDataLoss(correctProb, batchSize) + getRegulationLoss();

            // backpropagation step
            backPropagationStep(score, batch, batchLabels);

            // update data index
            dataIndex += batchLength;

            //clean
            delete batch;
            delete score;
            delete correctProb;
            delete[] batchLabels;
        }
        double lossMean = loss / numberOfBatches;
        std::cout << "epoch: " << e << " loss: " << lossMean << '\n';
    }
}