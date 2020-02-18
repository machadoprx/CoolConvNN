//
// Created by vmachado on 2/11/20.
//

#include "NeuralNet.h"

NeuralNet::NeuralNet() {};

NeuralNet::NeuralNet(int featuresDimension,
                    int outputDimension,
                    int additionalHiddenLayers,
                    int layersDimension,
                    int batchSize,
                    double learningRate){

    this->featuresDimension      = featuresDimension;
    this->outputDimension        = outputDimension;
    this->additionalHiddenLayers = additionalHiddenLayers;
    this->layersDimension        = layersDimension;
    this->batchSize              = batchSize;
    this->learningRate           = learningRate;

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
    fread(&batchSize, sizeof(int), 1, f);

    auto inputLayer = new Layer(featuresDimension, layersDimension, true);
    fread(inputLayer->weights->data, sizeof(double) * featuresDimension * layersDimension, 1, f);
    fread(inputLayer->gamma->data, sizeof(double) * layersDimension, 1, f);
    fread(inputLayer->beta->data, sizeof(double) * layersDimension, 1, f);
    fread(inputLayer->runningMean->data, sizeof(double) * layersDimension, 1, f);
    fread(inputLayer->runningVariance->data, sizeof(double) * layersDimension, 1, f);
    layers.push_back(inputLayer);

    for (int i = 0; i < additionalHiddenLayers; i++) {

        auto hiddenLayer = new Layer(layersDimension, layersDimension, true);
        fread(hiddenLayer->weights->data, sizeof(double) * layersDimension * layersDimension, 1, f);
        fread(hiddenLayer->gamma->data, sizeof(double) * layersDimension, 1, f);
        fread(hiddenLayer->beta->data, sizeof(double) * layersDimension, 1, f);
        fread(hiddenLayer->runningMean->data, sizeof(double) * layersDimension, 1, f);
        fread(hiddenLayer->runningVariance->data, sizeof(double) * layersDimension, 1, f);
        layers.push_back(hiddenLayer);

    }

    auto outputLayer = new Layer(layersDimension, outputDimension, false);
    fread(outputLayer->weights->data, sizeof(double) * layersDimension * outputDimension, 1, f);
    layers.push_back(outputLayer);

    fclose(f);
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
    fwrite(&batchSize, sizeof(int), 1, f);

    auto inputLayer = layers.at(0);
    fwrite(inputLayer->weights->data, sizeof(double) * featuresDimension * layersDimension, 1, f);
    fwrite(inputLayer->gamma->data, sizeof(double) * layersDimension, 1, f);
    fwrite(inputLayer->beta->data, sizeof(double) * layersDimension, 1, f);
    fwrite(inputLayer->runningMean->data, sizeof(double) * layersDimension, 1, f);
    fwrite(inputLayer->runningVariance->data, sizeof(double) * layersDimension, 1, f);

    int i = 1;
    for (; i < (int)layers.size() - 1; i++) {
        auto hiddenLayer = layers.at(i);
        fwrite(hiddenLayer->weights->data, sizeof(double) * layersDimension * layersDimension, 1, f);
        fwrite(hiddenLayer->gamma->data, sizeof(double) * layersDimension, 1, f);
        fwrite(hiddenLayer->beta->data, sizeof(double) * layersDimension, 1, f);
        fwrite(hiddenLayer->runningMean->data, sizeof(double) * layersDimension, 1, f);
        fwrite(hiddenLayer->runningVariance->data, sizeof(double) * layersDimension, 1, f);
    }

    auto outputLayer = layers.at(i);
    fwrite(outputLayer->weights->data, sizeof(double) * layersDimension * outputDimension, 1, f);

    fclose(f);

}

Matrix* NeuralNet::getCorrectProb(Matrix* prob, int const *labels){

    auto correctProb = new Matrix(prob->rows, 1);

    for (int i = 0; i < prob->rows; i++) {
        correctProb->data[i] = (-1) * log(prob->data[i * prob->columns + labels[i]]);
    }

    return correctProb;
}

Matrix* NeuralNet::getProbDerivative(Matrix* prob, int const *labels){

    int rows = prob->rows, columns = prob->columns;
    auto dProb = prob->copy();

    int stage = rows / THREADS;
    
    #pragma omp parallel for
    for (int t = 0; t < THREADS; t++) {
        int part = stage * t;
        for (int i = part; i < part + stage; i++) {
            int row = i * columns;
            dProb->data[i * columns + labels[i]] -= 1;

            for (int j = 0; j < columns; j++) {
                int index = row + j;
                dProb->data[index] = dProb->data[index] / rows;
            }
        }
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

    std::srand(std::time(nullptr));
    double *tmpPointer;
    int tmpLabel, randomIndex;

    for (int i = samples - 1; i >= 1; i--) {

        randomIndex = std::rand() % (i + 1);
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
            auto batchLabels = new int[batchLength];

            int x = 0;
            for (int i = dataIndex; i < dataIndex + batchLength; i++) {
                for (int j = 0; j < featuresDimension; j++) {
                    batch->data[x * featuresDimension + j] = dataSet[i][j];
                }
                batchLabels[x++] = labels[i];
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

void NeuralNet::setLearningRate(double value) {
    this->learningRate = value;
}