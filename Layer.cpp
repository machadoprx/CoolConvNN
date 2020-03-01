//
// Created by vmachado on 2/11/20.
//

#include <iostream>
#include "Layer.h"

Layer::Layer(int inputDimension, int outputDimension, bool hidden) {

    this->hidden = hidden;
    weights = new Matrix(inputDimension, outputDimension);

    if (hidden) {
        gamma = new Matrix(1, outputDimension);
        beta = new Matrix(1, outputDimension);
        runningMean = new Matrix(1, outputDimension);
        runningVariance = new Matrix(1, outputDimension);

        for (int j = 0; j < outputDimension; j++) {
            gamma->data[j] = 1.;
        }
    }

    weights->randomize();
}

Layer::Layer(int inputDimension, int outputDimension, bool hidden, float* weights, float* gamma, float* beta,
            float* runningMean, float* runningVariance) {

    this->hidden = hidden;
    this->weights = new Matrix(inputDimension, outputDimension);

    if (hidden) {
        this->gamma = new Matrix(1, outputDimension);
        this->beta = new Matrix(1, outputDimension);
        this->runningMean = new Matrix(1, outputDimension);
        this->runningVariance = new Matrix(1, outputDimension);

        #pragma omp parallel num_threads(THREADS)
        {
            #pragma omp for nowait
            for (int i = 0; i < outputDimension; i++) {
                this->gamma->data[i] = gamma[i];
                this->beta->data[i] = beta[i];
                this->runningMean->data[i] = runningMean[i];
                this->runningVariance->data[i] = runningVariance[i];
            }
        }
    }

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < inputDimension * outputDimension; i++) {
            this->weights->data[i] = weights[i];
        }
    }
}

Layer::~Layer() {
    delete weights;
    delete output;
    delete outputNormalized;
    delete gamma;
    delete beta;
    delete runningMean;
    delete runningVariance;
    delete deviationInv;
}

void Layer::validationOutput() {

    auto outputCentered = output->centralized(runningMean);

    delete deviationInv;
    deviationInv = Matrix::invDeviation(runningVariance);

    delete outputNormalized;
    outputNormalized = outputCentered->elemMulVector(deviationInv);

    delete output;
    output = outputNormalized->elemMulVector(gamma, beta);

    delete outputCentered;
}

void Layer::updateRunningStatus(Matrix* mean, Matrix* variance) {

    float momentum = 0.9;

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < mean->columns; i++) {
            runningMean->data[i] = (momentum * runningMean->data[i]) + (1 - momentum) * mean->data[i];
            runningVariance->data[i] = (momentum * runningVariance->data[i]) + (1 - momentum) * variance->data[i];
        }
    }


}

void Layer::trainingOutput() {

    auto mean = output->mean0Axis();
    auto variance = output->variance0Axis(); // do i should use centered output or untouched for variance calc
    auto outputCentered = output->centralized(mean);

    updateRunningStatus(mean, variance);

    delete deviationInv;
    deviationInv = Matrix::invDeviation(variance);

    delete outputNormalized;
    outputNormalized = outputCentered->elemMulVector(deviationInv);

    delete output;
    output = outputNormalized->elemMulVector(gamma, beta);

    delete outputCentered;
    delete mean;
    delete variance;
}

void Layer::feedForward(Matrix* input, bool validation){

    delete output;
    output = input->multiply(weights);

    if (hidden) {
        #pragma omp parallel num_threads(THREADS)
        {
            #pragma omp for nowait
            for (int i = 0; i < output->rows * output->columns; i++) {
                output->data[i] = Matrix::ReLU(output->data[i]);
            }
        }
        if (validation) {
            validationOutput();
        }
        else {
            trainingOutput();
        }
    }
}

Matrix* Layer::getBatchNormDerivative(Matrix* dOut, Layer* prev) {

    // https://kevinzakka.github.io/2016/09/14/batch_normalization/

    int rows = dOut->rows, columns = dOut->columns;
    auto oNormalized = prev->getOutputNormalized();
    auto prevDeviationInv = prev->getDeviationInv();
    auto prevGamma = prev->getGamma();

    auto dPart = new Matrix(rows, columns);
    auto dBatch0 = new Matrix(rows, columns);
    auto dBatch1 = new Matrix(rows, columns);
    float *dBatch2 = (float*) aligned_alloc(CACHE_LINE, sizeof(float) * columns);
    float *dBatch3 = (float*) aligned_alloc(CACHE_LINE, sizeof(float) * columns);

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < columns; i++) {
            dBatch2[i] = 0;
            dBatch3[i] = 0;
        }
    }

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {
            int index = i * columns;
            for (int j = 0; j < columns; j++) {
                dPart->data[index] = dOut->data[index] * prevGamma->data[j];
                dBatch1->data[index] = dPart->data[index] * rows;
                dBatch2[j] += dPart->data[index];
                dBatch3[j] += dPart->data[index] * oNormalized->data[index];
                index++;
            }
        }
    }

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < rows; i++) {
            int index = i * columns;
            for (int j = 0; j < columns; j++) {
                dBatch1->data[index] -= dBatch2[j];
                dBatch1->data[index] -= (oNormalized->data[index] * dBatch3[j]);
                dBatch0->data[index] = dBatch1->data[index] * prevDeviationInv->data[j] * (1.0 / (float) rows);
                index++;
            }
        }
    }

    delete oNormalized;
    delete prevDeviationInv;
    delete prevGamma;
    delete dBatch1;
    delete dPart;
    free(dBatch2);
    free(dBatch3);

    return dBatch0;
}

Matrix* Layer::backPropagation(Matrix *dOut, Matrix* &dWeights, Matrix* &dGamma, Matrix* &dBeta, Matrix *input, Layer* previous, float lambdaReg) {

    auto inputT = input->transposed();

    // Current weights derivative with L2 regularization
    auto dW = inputT->multiply(dOut);
    dWeights = dW->sum(weights, lambdaReg);

    delete inputT;
    delete dW;

    if(previous == nullptr) {
        return nullptr;
    }

    auto WT = weights->transposed();
    auto dInput = dOut->multiply(WT);

    // gamma and beta derivative for batch norm
    auto inputNorm = previous->getOutputNormalized();
    auto dGammaPartial = dInput->elemMul(inputNorm);
    dGamma = dGammaPartial->sumRows();
    dBeta = dInput->sumRows();

    // get input layer final derivative
    auto dInputNorm = getBatchNormDerivative(dInput, previous);

    // get relu derivative
    auto dInputReLU = dInputNorm->ReLUDerivative(input);

    //clear
    delete dGammaPartial;
    delete dInputNorm;
    delete inputNorm;
    delete dInput;
    delete WT;

    return dInputReLU;
}

void Layer::updateWeights(Matrix* dWeights, float learningRate){

    auto newWeights = weights->sum(dWeights, (-1) * learningRate);
    weights->set(newWeights);

    delete newWeights;
}

void Layer::updateGammaBeta(Matrix* dGamma, Matrix* dBeta, float learningRate) {

    assert(dGamma->rows == 1 && dBeta->rows == 1);
    assert ((gamma->columns == dGamma->columns ) && (beta->columns  == dBeta->columns));

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < dGamma->columns; i++) {
            gamma->data[i] -= learningRate * dGamma->data[i];
            beta->data[i] -= learningRate * dBeta->data[i];
        }
    }
}

Matrix* Layer::getOutput() {
    return output->copy();
}

Matrix* Layer::getWeights() {
    return weights->copy();
}

Matrix* Layer::getDeviationInv() {
    return deviationInv->copy();
}

Matrix* Layer::getOutputNormalized() {
    return outputNormalized->copy();
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
