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
                    float learningRate){

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
    fread(inputLayer->weights->data, sizeof(float) * featuresDimension * layersDimension, 1, f);
    fread(inputLayer->gamma->data, sizeof(float) * layersDimension, 1, f);
    fread(inputLayer->beta->data, sizeof(float) * layersDimension, 1, f);
    fread(inputLayer->runningMean->data, sizeof(float) * layersDimension, 1, f);
    fread(inputLayer->runningVariance->data, sizeof(float) * layersDimension, 1, f);
    layers.push_back(inputLayer);

    for (int i = 0; i < additionalHiddenLayers; i++) {

        auto hiddenLayer = new Layer(layersDimension, layersDimension, true);
        fread(hiddenLayer->weights->data, sizeof(float) * layersDimension * layersDimension, 1, f);
        fread(hiddenLayer->gamma->data, sizeof(float) * layersDimension, 1, f);
        fread(hiddenLayer->beta->data, sizeof(float) * layersDimension, 1, f);
        fread(hiddenLayer->runningMean->data, sizeof(float) * layersDimension, 1, f);
        fread(hiddenLayer->runningVariance->data, sizeof(float) * layersDimension, 1, f);
        layers.push_back(hiddenLayer);

    }

    auto outputLayer = new Layer(layersDimension, outputDimension, false);
    fread(outputLayer->weights->data, sizeof(float) * layersDimension * outputDimension, 1, f);
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
    fwrite(inputLayer->weights->data, sizeof(float) * featuresDimension * layersDimension, 1, f);
    fwrite(inputLayer->gamma->data, sizeof(float) * layersDimension, 1, f);
    fwrite(inputLayer->beta->data, sizeof(float) * layersDimension, 1, f);
    fwrite(inputLayer->runningMean->data, sizeof(float) * layersDimension, 1, f);
    fwrite(inputLayer->runningVariance->data, sizeof(float) * layersDimension, 1, f);

    int i = 1;
    for (; i < (int)layers.size() - 1; i++) {
        auto hiddenLayer = layers.at(i);
        fwrite(hiddenLayer->weights->data, sizeof(float) * layersDimension * layersDimension, 1, f);
        fwrite(hiddenLayer->gamma->data, sizeof(float) * layersDimension, 1, f);
        fwrite(hiddenLayer->beta->data, sizeof(float) * layersDimension, 1, f);
        fwrite(hiddenLayer->runningMean->data, sizeof(float) * layersDimension, 1, f);
        fwrite(hiddenLayer->runningVariance->data, sizeof(float) * layersDimension, 1, f);
    }

    auto outputLayer = layers.at(i);
    fwrite(outputLayer->weights->data, sizeof(float) * layersDimension * outputDimension, 1, f);

    fclose(f);

}

Matrix* NeuralNet::getCorrectProb(Matrix* prob, int *labels){

    auto correctProb = new Matrix(prob->rows, 1);

    for (int i = 0; i < prob->rows; i++) {
        correctProb->data[i] = (-1) * log(prob->data[i * prob->columns + labels[i]]);
    }

    return correctProb;
}

Matrix* NeuralNet::getProbDerivative(Matrix* prob, int *labels){

    int rows = prob->rows, columns = prob->columns;
    auto dProb = new Matrix(prob->rows, prob->columns);
    
    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {
            
            int row = i * columns;
            dProb->data[row + labels[i]] = -1.0;
            //#pragma omp for nowait // labels % THREADS must be 0?
            for (int j = 0; j < columns; j++) {
                int index = row + j;
                dProb->data[index] = (dProb->data[index] + prob->data[index]) / rows;
            }
        }
    }

    return dProb;
}

float NeuralNet::getDataLoss(Matrix* correctProb){

    float loss = 0;

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for reduction(+:loss)
        for (int i = 0; i < correctProb->rows; i++) {
            loss += correctProb->data[i];
        }
    }

    return loss / correctProb->rows;
}

float NeuralNet::getRegulationLoss(){

    float regLoss = 0;

    for (auto & layer : layers) {

        Matrix *weights = layer->getWeights();
        Matrix *w2 = weights->elemMul(weights);

        regLoss += 0.5 * lambdaReg * w2->sumElements();

        delete weights;
        delete w2;
    }

    return regLoss;
}

void NeuralNet::shuffleDataFisherYates(float** &data, int* labels, int samples) {

    //#pragma omp parallel num_threads(THREADS)
    {
        std::srand(int(time(NULL))); //^ omp_get_thread_num());
        //#pragma omp for
        for (int i = samples - 1; i >= 1; i--) {

            int randomIndex = std::rand() % (i + 1);
            float *tmpPointer = data[i];
            int tmpLabel = labels[i];

            data[i] = data[randomIndex];
            labels[i] = labels[randomIndex];

            data[randomIndex] = tmpPointer;
            labels[randomIndex] = tmpLabel;
        }
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

void NeuralNet::backPropagationStep(Matrix* prob, Matrix* batch, int *labels) {

    int layersSize = (int) layers.size() - 1;
    auto dCurrO = getProbDerivative(prob, labels);
    Layer *current, *previous;
    Matrix *dWeights, *dGamma, *dBeta;

    for (int i = layersSize; i >= 1; i--) {

        current = layers.at(i);
        previous = layers.at(i - 1);
        auto input = previous->getOutput();
        auto tmp = current->backPropagation(dCurrO, dWeights, dGamma, dBeta, input, previous, lambdaReg);

        // update current layer weights
        current->updateWeights(dWeights, learningRate);

        // update current layer weights
        previous->updateGammaBeta(dGamma, dBeta, learningRate);

        delete input;
        delete dCurrO;
        delete dGamma;
        delete dBeta;
        delete dWeights;
        dCurrO = tmp;
    }

    current = layers.at(0);
    current->backPropagation(dCurrO, dWeights, dGamma, dBeta, batch, nullptr, lambdaReg);
    current->updateWeights(dWeights, learningRate);

    delete dWeights;
    delete dCurrO;
}

void NeuralNet::prepareBatch(float** &dataSet, int* &labels, int batchLength, int dataIndex, Matrix *batch, int *batchLabels) {
    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < batchLength; i++) {
            for (int j = 0; j < featuresDimension; j++) {
                int index = i * featuresDimension + j;
                batch->data[index] = dataSet[dataIndex + i][j];
            }
            batchLabels[i] = labels[dataIndex + i];
        }
    }
}

void NeuralNet::train(float** &dataSet, int* &labels, int samples, int epochs){

    assert(samples >= batchSize);

    int numberOfBatches = samples % batchSize != 0 ?
                          (samples / batchSize) + 1
                          : samples / batchSize;

    for (int e = 1; e <= epochs; e++) {

        int dataIndex = 0;
        float loss = 0;

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

            prepareBatch(dataSet, labels, batchLength, dataIndex, batch, batchLabels);

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
        float lossMean = loss / numberOfBatches;
        std::cout << "epoch: " << e << " loss: " << lossMean << '\n';
    }
}

void NeuralNet::setLearningRate(float value) {
    this->learningRate = value;
}