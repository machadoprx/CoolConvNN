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

    layers.push_back(new Layer(featuresDimension, layersDimension, true));
    for (int i = 0; i < additionalHiddenLayers; i++) {
        layers.push_back(new Layer(layersDimension, layersDimension, true));
    }
    layers.push_back(new Layer(layersDimension, outputDimension, false));
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

    auto inputLayer = new Layer(featuresDimension, layersDimension, true);
    fread(inputLayer->getWeights()->data, sizeof(double) * featuresDimension * layersDimension, 1, f);
    fread(inputLayer->getGamma()->data, sizeof(double) * layersDimension, 1, f);
    fread(inputLayer->getBeta()->data, sizeof(double) * layersDimension, 1, f);
    fread(inputLayer->getRunningMean()->data, sizeof(double) * layersDimension, 1, f);
    fread(inputLayer->getRunningVariance()->data, sizeof(double) * layersDimension, 1, f);
    layers.push_back(inputLayer);

    for (int i = 0; i < additionalHiddenLayers; i++) {

        auto *hiddenLayer = new Layer(layersDimension, layersDimension, true);
        fread(hiddenLayer->getWeights()->data, sizeof(double) * layersDimension * layersDimension, 1, f);
        fread(hiddenLayer->getGamma()->data, sizeof(double) * layersDimension, 1, f);
        fread(hiddenLayer->getBeta()->data, sizeof(double) * layersDimension, 1, f);
        fread(hiddenLayer->getRunningMean()->data, sizeof(double) * layersDimension, 1, f);
        fread(hiddenLayer->getRunningVariance()->data, sizeof(double) * layersDimension, 1, f);
        layers.push_back(hiddenLayer);

    }

    auto outputLayer = new Layer(layersDimension, outputDimension, false);
    fread(outputLayer->getWeights()->data, sizeof(double) * layersDimension * outputDimension, 1, f);
    fread(outputLayer->getGamma()->data, sizeof(double) * outputDimension, 1, f);
    fread(outputLayer->getBeta()->data, sizeof(double) * outputDimension, 1, f);
    fread(outputLayer->getRunningMean()->data, sizeof(double) * outputDimension, 1, f);
    fread(outputLayer->getRunningVariance()->data, sizeof(double) * outputDimension, 1, f);
    layers.push_back(outputLayer);

}

NeuralNet::~NeuralNet() {
    for (auto & layer : layers) {
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

    auto inputLayer = layers.at(0);
    fwrite(inputLayer->getWeights()->data, sizeof(double) * featuresDimension * layersDimension, 1, f);
    fwrite(inputLayer->getGamma()->data, sizeof(double) * layersDimension, 1, f);
    fwrite(inputLayer->getBeta()->data, sizeof(double) * layersDimension, 1, f);
    fwrite(inputLayer->getRunningMean()->data, sizeof(double) * layersDimension, 1, f);
    fwrite(inputLayer->getRunningVariance()->data, sizeof(double) * layersDimension, 1, f);

    int i = 1;
    for (; i < (int)layers.size() - 1; i++) {
        auto hiddenLayer = layers.at(i);
        fwrite(hiddenLayer->getWeights()->data, sizeof(double) * layersDimension * layersDimension, 1, f);
        fwrite(hiddenLayer->getGamma()->data, sizeof(double) * layersDimension, 1, f);
        fwrite(hiddenLayer->getBeta()->data, sizeof(double) * layersDimension, 1, f);
        fwrite(hiddenLayer->getRunningMean()->data, sizeof(double) * layersDimension, 1, f);
        fwrite(hiddenLayer->getRunningVariance()->data, sizeof(double) * layersDimension, 1, f);
    }

    auto outputLayer = layers.at(i);
    fwrite(outputLayer->getWeights()->data, sizeof(double) * layersDimension * outputDimension, 1, f);
    fwrite(outputLayer->getGamma()->data, sizeof(double) * outputDimension, 1, f);
    fwrite(outputLayer->getBeta()->data, sizeof(double) * outputDimension, 1, f);
    fwrite(outputLayer->getRunningMean()->data, sizeof(double) * outputDimension, 1, f);
    fwrite(outputLayer->getRunningVariance()->data, sizeof(double) * outputDimension, 1, f);

}

Matrix* NeuralNet::getCorrectProb(Matrix* prob, int const *labels){

    auto correctProb = new Matrix(prob->rows, 1);

    for (int i = 0; i < prob->rows; i++) {
        correctProb->data[i] = (-1) * log(prob->data[i * prob->columns + labels[i]]);
    }

    return correctProb;
}

Matrix* NeuralNet::getProbDerivative(Matrix* prob, int const *labels){

    int rows = prob->rows, columns = prob->columns, currRow = 0;
    auto dProb = prob->copy();

    for (int i = 0; i < rows * columns; i++) {

        if (i % columns == 0) {
            dProb->data[i + labels[currRow]] -= 1;
            currRow++;
        }

        dProb->data[i] = dProb->data[i] / rows;
    }

    return dProb;
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

        regLoss += 0.5 * lambdaReg * w2->sumElements();

        delete weights;
        delete w2;
    }

    return regLoss;
}

void NeuralNet::shuffleDataFisherYates(double** &data, int* labels, int samples) {

    srand(time(NULL));
    double *tmpPointer;
    //auto tmpData = new double[featuresDimension];
    int tmpLabel, randomIndex;

    for (int i = samples - 1; i >= 1; i--) {

        randomIndex = rand() % (i + 1);
        tmpPointer = data[i];
        tmpLabel = labels[i];

        data[i] = data[randomIndex];
        labels[i] = labels[randomIndex];

        data[randomIndex] = tmpPointer;
        labels[randomIndex] = tmpLabel;
    }
}

Matrix* NeuralNet::forwardStep(Matrix* batch, bool validation) {

    int layersSize = (int)layers.size() - 1;
    Layer *current = layers.at(0), *previous;
    current->feedForward(batch, validation);

    for (int i = 1; i < layersSize; i++) {
        previous = current;
        current = layers.at(i);
        auto prevO = previous->getOutput();
        current->feedForward(prevO, validation);
        delete prevO;
    }

    previous = current;
    auto prevO = previous->getOutput();
    current = layers.at(layersSize);
    current->feedForward(prevO, validation);
    delete prevO;

    auto out = current->getOutput();
    auto prob = out->normalized();
    delete out;

    return prob;
}

void NeuralNet::backPropagationStep(Matrix* prob, Matrix* batch, int const *labels) {

    int layersSize = (int) layers.size() - 1;
    auto dCurrO = getProbDerivative(prob, labels);
    Layer *current, *previous;

    for (int i = layersSize; i >= 1; i--) {

        current = layers.at(i);
        previous = layers.at(i - 1);
        auto input = previous->getOutput();
        auto tmp = current->backPropagation(dCurrO, input, previous, learningRate, lambdaReg);

        delete input;
        delete dCurrO;

        dCurrO = tmp;
    }

    current = layers.at(0);

    current->backPropagation(dCurrO, batch, nullptr, learningRate, lambdaReg);

    delete dCurrO;
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
            int batchLength = batchSize;

            if (dataIndex + batchSize >= samples){
                batchLength = samples - dataIndex;
            }

            auto batch = new Matrix(batchLength, featuresDimension);
            int *batchLabels = new int[batchLength];

            int x = 0;
            for (int i = dataIndex; i < dataIndex + batchLength; i++) {
                for (int j = 0; j < featuresDimension; j++) {
                    batch->data[x * featuresDimension + j] = dataSet[i][j];
                }
                batchLabels[x] = labels[i];
                x++;
            }

            //forward step
            auto score = forwardStep(batch, false);

            // get correct probabilities for each class
            auto correctProb = getCorrectProb(score, batchLabels);

            // compute loss
            // L = 1/n * sum (loss) + for each layer(1/2 * lambda * sum (layer weight @ layer weight))
            loss += getDataLoss(correctProb) + getRegulationLoss();

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
        double lossMean = loss / numberOfBatches;
        std::cout << "epoch: " << e << " loss: " << lossMean << '\n';
    }
}