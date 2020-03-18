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

    layers.push_back(new Layer(featuresDimension, layersDimension, true, true));
    for (int i = 0; i < additionalHiddenLayers; i++) {
        layers.push_back(new Layer(layersDimension, layersDimension, true, false));
    }
    layers.push_back(new Layer(layersDimension, outputDimension, false, false));
}

NeuralNet::NeuralNet(const char *path) { // array alignment

    FILE *f = fopen(path, "rb");

    if (f == nullptr) {
        throw std::runtime_error("invalid file");
    }

    float *weights = nullptr, *gamma = nullptr, *beta = nullptr, *runningMean = nullptr, *runningVariance = nullptr;

    fread(&additionalHiddenLayers, sizeof(int), 1, f);
    fread(&featuresDimension, sizeof(int), 1, f);
    fread(&layersDimension, sizeof(int), 1, f);
    fread(&outputDimension, sizeof(int), 1, f);
    fread(&batchSize, sizeof(int), 1, f);

    weights = new float[featuresDimension * layersDimension];
    fread(weights, sizeof(float) * layersDimension * featuresDimension, 1, f);
    layers.push_back(new Layer(featuresDimension, layersDimension, true, true, weights, gamma, beta,
                    runningMean, runningVariance));

    delete weights;

    for (int i = 0; i < additionalHiddenLayers; i++) {
        
        weights = new float[layersDimension * layersDimension];
        gamma = new float[layersDimension];
        beta = new float[layersDimension];
        runningMean = new float[layersDimension];
        runningVariance = new float[layersDimension];

        fread(weights, sizeof(float) * layersDimension * layersDimension, 1, f);
        fread(gamma, sizeof(float) * layersDimension, 1, f);
        fread(beta, sizeof(float) * layersDimension, 1, f);
        fread(runningMean, sizeof(float) * layersDimension, 1, f);
        fread(runningVariance, sizeof(float) * layersDimension, 1, f);
        layers.push_back(new Layer(layersDimension, layersDimension, true, false, weights, gamma, beta,
                                runningMean, runningVariance));
        delete weights;
        delete gamma;
        delete beta;
        delete runningMean;
        delete runningVariance;
    }

    weights = new float[outputDimension * layersDimension];
    gamma = new float[layersDimension];
    beta = new float[layersDimension];
    runningMean = new float[layersDimension];
    runningVariance = new float[layersDimension];

    fread(weights, sizeof(float) * layersDimension * outputDimension, 1, f);
    fread(gamma, sizeof(float) * layersDimension, 1, f);
    fread(beta, sizeof(float) * layersDimension, 1, f);
    fread(runningMean, sizeof(float) * layersDimension, 1, f);
    fread(runningVariance, sizeof(float) * layersDimension, 1, f);
    layers.push_back(new Layer(layersDimension, outputDimension, false, false, weights, gamma, beta,
                                runningMean, runningVariance));

    delete weights;
    delete gamma;
    delete beta;
    delete runningMean;
    delete runningVariance;

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

    Layer *l;
    Matrix *weights, *gamma, *beta, *runningMean, *runningVariance;

    fwrite(&additionalHiddenLayers, sizeof(int), 1, f);
    fwrite(&featuresDimension, sizeof(int), 1, f);
    fwrite(&layersDimension, sizeof(int), 1, f);
    fwrite(&outputDimension, sizeof(int), 1, f);
    fwrite(&batchSize, sizeof(int), 1, f);

    l = layers.at(0);
    weights = l->getWeights();
    fwrite(weights->data, sizeof(float) * featuresDimension * layersDimension, 1, f);
    delete weights;

    int i = 1;
    for (; i < (int) layers.size() - 1; i++) {
        l = layers.at(i);
        
        weights = l->getWeights();
        gamma = l->getGamma();
        beta = l->getBeta();
        runningMean = l->getRunningMean();
        runningVariance = l->getRunningVariance();

        fwrite(weights->data, sizeof(float) * layersDimension * layersDimension, 1, f);
        fwrite(gamma->data, sizeof(float) * layersDimension, 1, f);
        fwrite(beta->data, sizeof(float) * layersDimension, 1, f);
        fwrite(runningMean->data, sizeof(float) * layersDimension, 1, f);
        fwrite(runningVariance->data, sizeof(float) * layersDimension, 1, f);

        delete weights;
        delete gamma;
        delete beta;
        delete runningMean;
        delete runningVariance;
    }

    l = layers.at(i);

    weights = l->getWeights();
    gamma = l->getGamma();
    beta = l->getBeta();
    runningMean = l->getRunningMean();
    runningVariance = l->getRunningVariance();

    fwrite(weights->data, sizeof(float) * layersDimension * outputDimension, 1, f);
    fwrite(gamma->data, sizeof(float) * layersDimension, 1, f);
    fwrite(beta->data, sizeof(float) * layersDimension, 1, f);
    fwrite(runningMean->data, sizeof(float) * layersDimension, 1, f);
    fwrite(runningVariance->data, sizeof(float) * layersDimension, 1, f);

    delete weights;
    delete gamma;
    delete beta;
    delete runningMean;
    delete runningVariance;

    fclose(f);

}

Matrix* NeuralNet::getCorrectProb(Matrix* prob, int *labels){

    auto correctProb = new Matrix(prob->rows, 1);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < prob->rows; i++) {
            correctProb->data[i] = (-1.0f) * logf(prob->data[i * prob->columns + labels[i]]);
        }
    }

    return correctProb;
}

Matrix* NeuralNet::getProbDerivative(Matrix* prob, int *labels){

    int rows = prob->rows, columns = prob->columns;
    auto dProb = new Matrix(prob->rows, prob->columns);
    
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {
            
            int index = i * columns;
            dProb->data[index + labels[i]] = -1.0f;

            for (int j = 0; j < columns; j++) {
                dProb->data[index] = (dProb->data[index] + prob->data[index]) / (float)rows;
                index++;
            }
        }
    }

    return dProb;
}

float NeuralNet::getDataLoss(Matrix* correctProb){

    float loss = .0f;

    #pragma omp parallel
    {
        #pragma omp for reduction (+:loss) schedule(static)
        for (int i = 0; i < correctProb->rows; i++) {
            loss += correctProb->data[i];
        }
    }

    return loss / correctProb->rows;
}

float NeuralNet::getRegulationLoss(std::vector<Layer*> la, float lambda){

    float regLoss = .0f;

    for (auto & layer : la) {
        auto weights = layer->getWeights();
        auto temp = weights->elemMul(weights);
        regLoss += 0.5f * lambda * temp->sumElements();
        delete weights;
        delete temp;
    }

    return regLoss;
}

void NeuralNet::shuffleDataFisherYates(float** &data, int* labels, int samples) {

    int randomIndex, tmpLabel;
    float *tmpPointer;
    std::srand(int(time(NULL)));
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

    int layersSize = (int) layers.size() - 1;
    auto curr = layers.at(0)->feedForward(batch, validation);

    for (int i = 1; i <= layersSize; i++) {
        auto temp = layers.at(i)->feedForward(curr, validation);
        delete curr;
        curr = temp;
    }

    auto prob = curr->normalized();
    delete curr;

    return prob;
}

void NeuralNet::backPropagationStep(Matrix* prob, Matrix* batch, int *labels) {

    int layersSize = (int) layers.size() - 1;
    auto dOut = getProbDerivative(prob, labels);

    for (int i = layersSize; i >= 1; i--) {
        auto tmp = layers.at(i)->backPropagation(dOut, lambdaReg, learningRate);

        delete dOut;
        dOut = tmp;
    }
    layers.at(0)->backPropagation(dOut, lambdaReg, learningRate);

    delete dOut;
}

void NeuralNet::prepareBatch(float** &dataSet, int* &labels, int batchLength, int dataIndex, Matrix *batch, int *batchLabels, int dataDim) {
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < batchLength; i++) {
            int index = i * dataDim;
            for (int j = 0; j < dataDim; j++) {
                batch->data[index] = dataSet[dataIndex + i][j];
                index++;
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
            int batchLength;
            if (dataIndex + batchSize >= samples) {
                batchLength = samples - dataIndex;
            }
            else {
                batchLength = batchSize;
            }
            auto batch = new Matrix(batchLength, featuresDimension);
            auto batchLabels = new int[batchLength];

            prepareBatch(dataSet, labels, batchLength, dataIndex, batch, batchLabels, featuresDimension);

            //forward step
            auto score = forwardStep(batch, false);

            // get correct probabilities for each class
            auto correctProb = getCorrectProb(score, batchLabels);

            // compute loss
            loss += getDataLoss(correctProb) + getRegulationLoss(layers, lambdaReg);

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

void NeuralNet::setLearningRate(float value) {
    this->learningRate = value;
}