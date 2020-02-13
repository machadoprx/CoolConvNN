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

    layers.push_back(new Layer(featuresDimension, layersDimension));
    for (int i = 0; i < additionalHiddenLayers; i++) {
        layers.push_back(new Layer(layersDimension, layersDimension));
    }
    layers.push_back(new Layer(layersDimension, outputDimension));
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
    fread(inputLayer->weights->data, sizeof(double) * featuresDimension * layersDimension, 1, f);
    fread(inputLayer->gamma->data, sizeof(double) * layersDimension, 1, f);
    fread(inputLayer->beta->data, sizeof(double) * layersDimension, 1, f);
    fread(inputLayer->runningMean->data, sizeof(double) * layersDimension, 1, f);
    fread(inputLayer->runningVariance->data, sizeof(double) * layersDimension, 1, f);
    layers.push_back(inputLayer);

    for (int i = 0; i < additionalHiddenLayers; i++) {

        auto *hiddenLayer = new Layer(layersDimension, layersDimension);
        fread(hiddenLayer->weights->data, sizeof(double) * layersDimension * layersDimension, 1, f);
        fread(hiddenLayer->gamma->data, sizeof(double) * layersDimension, 1, f);
        fread(hiddenLayer->beta->data, sizeof(double) * layersDimension, 1, f);
        fread(hiddenLayer->runningMean->data, sizeof(double) * layersDimension, 1, f);
        fread(hiddenLayer->runningVariance->data, sizeof(double) * layersDimension, 1, f);
        layers.push_back(hiddenLayer);

    }

    auto *outputLayer = new Layer(layersDimension, outputDimension);
    fread(outputLayer->weights->data, sizeof(double) * layersDimension * outputDimension, 1, f);
    fread(outputLayer->gamma->data, sizeof(double) * outputDimension, 1, f);
    fread(outputLayer->beta->data, sizeof(double) * outputDimension, 1, f);
    fread(outputLayer->runningMean->data, sizeof(double) * outputDimension, 1, f);
    fread(outputLayer->runningVariance->data, sizeof(double) * outputDimension, 1, f);
    layers.push_back(outputLayer);

}

NeuralNet::~NeuralNet() {
    for (auto* layer : layers) {
        delete layer;
    }
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
    fwrite(inputLayer->weights->data, sizeof(double) * featuresDimension * layersDimension, 1, f);
    fwrite(inputLayer->gamma->data, sizeof(double) * layersDimension, 1, f);
    fwrite(inputLayer->beta->data, sizeof(double) * layersDimension, 1, f);
    fwrite(inputLayer->runningMean->data, sizeof(double) * layersDimension, 1, f);
    fwrite(inputLayer->runningVariance->data, sizeof(double) * layersDimension, 1, f);

    int i = 1;
    for (; i < layers.size() - 1; i++) {
        auto *hiddenLayer = layers.at(i);
        fwrite(hiddenLayer->weights->data, sizeof(double) * layersDimension * layersDimension, 1, f);
        fwrite(hiddenLayer->gamma->data, sizeof(double) * layersDimension, 1, f);
        fwrite(hiddenLayer->beta->data, sizeof(double) * layersDimension, 1, f);
        fwrite(hiddenLayer->runningMean->data, sizeof(double) * layersDimension, 1, f);
        fwrite(hiddenLayer->runningVariance->data, sizeof(double) * layersDimension, 1, f);
    }

    auto *outputLayer = layers.at(i);
    fwrite(outputLayer->weights->data, sizeof(double) * layersDimension * outputDimension, 1, f);
    fwrite(outputLayer->gamma->data, sizeof(double) * outputDimension, 1, f);
    fwrite(outputLayer->beta->data, sizeof(double) * outputDimension, 1, f);
    fwrite(outputLayer->runningMean->data, sizeof(double) * outputDimension, 1, f);
    fwrite(outputLayer->runningVariance->data, sizeof(double) * outputDimension, 1, f);

}

Matrix* NeuralNet::getCorrectProb(Matrix* prob, const int *labels){

    auto *correctProb = new Matrix(prob->rows, 1);

    for (int i = 0; i < prob->rows; i++) {
        correctProb->data[i] = (-1) * log(prob->data[i * prob->columns + labels[i]]);
    }

    return correctProb;
}

Matrix* NeuralNet::getProbDerivative(Matrix* prob, const int* labels){

    int rows = prob->rows, columns = prob->columns, currRow = 0;
    auto *dProb = prob->copy();

    for (int i = 0; i < rows * columns; i++) {

        if (i % columns == 0) {
            dProb->data[i + labels[currRow]] -= 1;
            currRow++;
        }

        dProb->data[i] = dProb->data[i] / rows;
    }

    return dProb;
}

Matrix* NeuralNet::getReLUDerivative(Matrix* W, Matrix* W1) {

    int rows = W->rows, columns = W->columns;
    auto *R = new Matrix(W->rows, W->columns);

    for (int i = 0; i < rows * columns; i++) {

        R->data[i] = W1->data[i] <= 0 ? 0 : W->data[i];

    }

    return R;
}

double NeuralNet::getDataLoss(Matrix* correctProb){

    double loss = 0;

    for (int i = 0; i < correctProb->rows; i++) {
        loss += correctProb->data[i];
    }
    loss = loss / correctProb->rows;

    return loss;
}

double NeuralNet::getRegulationLoss(){

    double regLoss = 0;

    for (auto & layer : layers) {
        Matrix *weights = layer->getWeights();
        Matrix *w2 = weights->elemMul(weights);
        double t = w2->sumElements();
        regLoss += 0.5 * lambdaReg * t;
        delete weights;
        delete w2;
    }

    return regLoss;
}

void NeuralNet::shuffleDataFisherYates(double** &data, int* &labels, int samples) {

    srand(time(NULL));
    auto *tmpData = new double[featuresDimension];
    int tmpLabel, randomIndex;

    for (int i = samples - 1; i >= 1; i--) {

        randomIndex = rand() % (i + 1);
        memcpy(tmpData, data[i], featuresDimension * sizeof(double));
        tmpLabel = labels[i];

        memcpy(data[i], data[randomIndex], featuresDimension * sizeof(double));
        labels[i] = labels[randomIndex];

        memcpy(data[randomIndex], tmpData, featuresDimension * sizeof(double));
        labels[randomIndex] = tmpLabel;
    }

    delete[] tmpData;
}

Matrix* NeuralNet::getBatchNormDerivative(Matrix* dOut, Layer* layer) {

    // https://kevinzakka.github.io/2016/09/14/batch_normalization/

    int rows = dOut->rows, columns = dOut->columns;
    auto *oNormalized = layer->getOutputNormalized();
    auto *deviationInv = layer->getDeviationInv();
    auto *gamma = layer->getGamma();

    auto *dBatch0 = new Matrix(dOut->rows, dOut->columns);
    auto *dBatch1 = new Matrix(dOut->rows, dOut->columns);
    auto *dBatch2 = new double[dOut->columns];
    auto *dBatch3 = new double[dOut->columns];
    memset(dBatch2, 0, sizeof(double) * columns);
    memset(dBatch3, 0, sizeof(double) * columns);

    for (int i = 0; i < rows * columns; i++) {

        int j = i % columns;
        double elem = dOut->data[i] * gamma->data[j];
        dBatch1->data[i] = elem * rows;
        dBatch2[j] += elem;
        dBatch3[j] += elem * oNormalized->data[i];

    }

    for (int i = 0; i < rows * columns; i++) {

        int j = i % columns;
        dBatch1->data[i] = dBatch1->data[i] - dBatch2[j];
        dBatch1->data[i] = dBatch1->data[i] - (oNormalized->data[i] * dBatch3[j]);
        dBatch0->data[i] = dBatch1->data[i] * deviationInv->data[j] * (1.0 / (double) rows);
    }

    delete oNormalized;
    delete deviationInv;
    delete gamma;
    delete dBatch1;
    delete[] dBatch2;
    delete[] dBatch3;

    return dBatch0;
}

Matrix* NeuralNet::forwardStep(Matrix* batch, bool validation) {

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

void NeuralNet::backPropagationStep(Matrix* prob, Matrix* batch, int* labels) {

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

        // gamma and beta derivative for batch norm
        Matrix *dPrevO = dCurrO->multiply(currWT);
        Matrix *prevONorm = previous->getOutputNormalized();
        Matrix *dGammaPartial = dPrevO->elemMul(prevONorm);
        Matrix *dGamma = dGammaPartial->sumRows();
        Matrix *dBeta = dPrevO->sumRows();

        // update previous layer batch norm weights
        previous->updateGammaBeta(dGamma, dBeta, learningRate);

        // get prev layer output final derivative
        Matrix *dPrevONorm = getBatchNormDerivative(dPrevO, previous);

        // update current layer weights
        current->updateWeights(dCurrWReg, learningRate);

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

void NeuralNet::train(double** &dataSet, int* &labels, int samples, int epochs){

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

            int x = 0;
            for (int i = dataIndex; i < dataIndex + batchLength; i++) {
                for (int j = 0; j < featuresDimension; j++) {
                    batch->data[x * featuresDimension + j] = dataSet[i][j];
                }
                batchLabels[x] = labels[i];
                x++;
            }

            //forward step
            Matrix *score = forwardStep(batch, false);

            // get correct probabilities for each class
            Matrix *correctProb = getCorrectProb(score, batchLabels);

            // compute loss
            // L = 1/n * sum (loss) + for each layer(1/2 * lambda * sum (layer weight @ layer weight))
            loss += getDataLoss(correctProb) + getRegulationLoss();

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