//
// Created by vmachado on 2/11/20.
//

#include "Layer.h"

Layer::Layer(int inputDimension, int outputDimension, bool hidden, bool isFirst) {

    this->hidden = hidden;
    this->isFirst = isFirst;
    weights = new Matrix(inputDimension, outputDimension);

    if (!isFirst) {
        gamma = new Matrix(1, inputDimension);
        beta = new Matrix(1, inputDimension);
        runningMean = new Matrix(1, inputDimension);
        runningVariance = new Matrix(1, inputDimension);

        for (int j = 0; j < inputDimension; j++) {
            gamma->data[j] = 1.f;
        }
    }

    weights->randomize(0.0f, sqrtf(2.0f / (float)inputDimension));
}

Layer::Layer(int inputDimension, int outputDimension, bool hidden, bool isFirst, float* weights, float* gamma, float* beta,
            float* runningMean, float* runningVariance) {

    this->hidden = hidden;
    this->weights = new Matrix(inputDimension, outputDimension);
    this->isFirst = isFirst;

    if (!isFirst) {
        this->gamma = new Matrix(1, inputDimension);
        this->beta = new Matrix(1, inputDimension);
        this->runningMean = new Matrix(1, inputDimension);
        this->runningVariance = new Matrix(1, inputDimension);

        for (int i = 0; i < inputDimension; i++) {
            this->gamma->data[i] = gamma[i];
            this->beta->data[i] = beta[i];
            this->runningMean->data[i] = runningMean[i];
            this->runningVariance->data[i] = runningVariance[i];
        }
    }

    for (int i = 0; i < inputDimension * outputDimension; i++) {
        this->weights->data[i] = weights[i];
    }
}

Layer::~Layer() {
    delete weights;
    delete input;
    delete inputNormalized;
    delete gamma;
    delete beta;
    delete runningMean;
    delete runningVariance;
    delete deviationInv;
}

void Layer::updateRunningStatus(Matrix* mean, Matrix* variance) {

    float momentum = 0.9f;
    float nmomentum = 1.0f - momentum;

    #pragma omp parallel for
    for (int i = 0; i < mean->columns; i++) {
        runningMean->data[i] = (momentum * runningMean->data[i]) + (nmomentum * mean->data[i]);
        runningVariance->data[i] = (momentum * runningVariance->data[i]) + (nmomentum * variance->data[i]);
    }
}

Matrix* Layer::feedForward(Matrix* rawInput, bool validation){

    delete input;

    if (!isFirst) {

        delete deviationInv;
        delete inputNormalized;

        if (!validation) {
            auto mean = rawInput->mean0Axis();
            auto variance = rawInput->variance0Axis(mean);
            updateRunningStatus(mean, variance);
            deviationInv = Matrix::invDeviation(variance);
            inputNormalized = rawInput->normalized2(mean, deviationInv);

            delete mean;
            delete variance;
        }
        
        else {
            deviationInv = Matrix::invDeviation(runningVariance);
            inputNormalized = rawInput->normalized2(runningMean, deviationInv);
        }

        input = inputNormalized->elemMulVector(gamma, beta);
    }
    else {
        input = rawInput->copy();
    }

    auto output = input->multiply(weights, false, false);

    if (hidden) {
        output->apply_relu();
    }

    return output;
}

Matrix* Layer::getBatchNormDerivative(Matrix* dInput) {

    // https://kevinzakka.github.io/2016/09/14/batch_normalization/

    int rows = dInput->rows, columns = dInput->columns;

    auto R = new Matrix(rows, columns);

    float *dPart1 = (float*) aligned_alloc(CACHE_LINE, sizeof(float) * columns);
    float *dPart2 = (float*) aligned_alloc(CACHE_LINE, sizeof(float) * columns);

    float rowsInv = 1.0f / (float)rows;

    for (int i = 0; i < columns; i++) {
        dPart1[i] = .0f;
        dPart2[i] = .0f;
    }

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int index = i * columns;
        float elem;
        for (int j = 0; j < columns; j++) {
            elem = dInput->data[index] * gamma->data[j];
            dPart1[j] += elem;
            dPart2[j] += elem * inputNormalized->data[index];
            R->data[index] = elem * rows;
            index++;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int index = i * columns;
        for (int j = 0; j < columns; j++) {
            R->data[index] -= dPart1[j];
            R->data[index] -= (inputNormalized->data[index] * dPart2[j]);
            R->data[index] = R->data[index] * (deviationInv->data[j] * rowsInv);
            index++;
        }
    }

    free(dPart1);
    free(dPart2);

    return R;
}

Matrix* Layer::backPropagation(Matrix *dOut, float lambdaReg, float learningRate) {

    // Current weights derivative with L2 regularization
    auto dWeights = input->multiply(dOut, true, false);
    dWeights->apply_sum(weights, lambdaReg);

    auto dInput = dOut->multiply(weights, false, true);

    // gamma and beta derivative for batch norm
    auto dGammaPartial = dInput->elemMul(inputNormalized);
    auto dGamma = dGammaPartial->sumRows();
    auto dBeta = dInput->sumRows();
    
    // get input layer final derivative
    auto dInputNorm = getBatchNormDerivative(dInput);

    // get relu derivative
    dInputNorm->apply_reluderivative(input);

    // update current layer weights
    updateWeights(dWeights, learningRate);
    updateGammaBeta(dGamma, dBeta, learningRate);

    //clear
    delete dWeights;
    delete dGamma;
    delete dBeta;
    delete dGammaPartial;
    delete dInput;

    return dInputNorm;
}

void Layer::updateWeights(Matrix* dWeights, float learningRate){
    weights->apply_sum(dWeights, (-1.0f) * learningRate);
}

void Layer::updateGammaBeta(Matrix* dGamma, Matrix* dBeta, float learningRate) {
    gamma->apply_sum(dGamma, (-1.0f) * learningRate);
    beta->apply_sum(dBeta, (-1.0f) * learningRate);
}

Matrix* Layer::getWeights() {
    return weights->copy();
}

Matrix* Layer::getDeviationInv() {
    return deviationInv->copy();
}

Matrix* Layer::getGamma() {
    return gamma->copy();
}

Matrix* Layer::getBeta() {
    return beta->copy();
}

Matrix* Layer::getRunningMean() {
    return runningMean->copy();
}

Matrix* Layer::getRunningVariance() {
    return runningVariance->copy();
}
