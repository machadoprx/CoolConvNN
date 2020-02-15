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

Layer::Layer(int inputDimension, int outputDimension, bool hidden, double* weights, double* gamma, double* beta,
            double* runningMean, double* runningVariance) {

    this->hidden = hidden;
    this->weights = new Matrix(inputDimension, outputDimension);

    if (hidden) {
        this->gamma = new Matrix(1, outputDimension);
        this->beta = new Matrix(1, outputDimension);
        this->runningMean = new Matrix(1, outputDimension);
        this->runningVariance = new Matrix(1, outputDimension);

        memcpy(this->gamma->data, gamma, sizeof(double) * outputDimension);
        memcpy(this->beta->data, beta, sizeof(double) * outputDimension);
        memcpy(this->runningMean->data, runningMean, sizeof(double) * outputDimension);
        memcpy(this->runningVariance->data, runningVariance, sizeof(double) * outputDimension);
    }

    memcpy(this->weights->data, weights, sizeof(double) * inputDimension * outputDimension);
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

double Layer::ReLU(double x){
    return x > 0 ? x : 0;
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

    double momentum = 0.9;

    for (int i = 0; i < mean->columns; i++) {
        runningMean->data[i] = (momentum * runningMean->data[i]) + (1 - momentum) * mean->data[i];
        runningVariance->data[i] = (momentum * runningVariance->data[i]) + (1 - momentum) * variance->data[i];
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

        int len = output->rows * output->columns;

        for (int i = 0; i < len; i++) {
            output->data[i] = ReLU(output->data[i]);
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

    auto dBatch0 = new Matrix(dOut->rows, dOut->columns);
    auto dBatch1 = new Matrix(dOut->rows, dOut->columns);
    auto dBatch2 = new double[dOut->columns];
    auto dBatch3 = new double[dOut->columns];
    memset(dBatch2, 0, sizeof(double) * columns);
    memset(dBatch3, 0, sizeof(double) * columns);

    for (int i = 0; i < rows * columns; i++) {

        int j = i % columns;
        double elem = dOut->data[i] * prevGamma->data[j];
        dBatch1->data[i] = elem * rows;
        dBatch2[j] += elem;
        dBatch3[j] += elem * oNormalized->data[i];

    }

    for (int i = 0; i < rows * columns; i++) {

        int j = i % columns;
        dBatch1->data[i] = dBatch1->data[i] - dBatch2[j];
        dBatch1->data[i] = dBatch1->data[i] - (oNormalized->data[i] * dBatch3[j]);
        dBatch0->data[i] = dBatch1->data[i] * prevDeviationInv->data[j] * (1.0 / (double) rows);
    }

    delete oNormalized;
    delete prevDeviationInv;
    delete prevGamma;
    delete dBatch1;
    delete[] dBatch2;
    delete[] dBatch3;

    return dBatch0;
}

Matrix* Layer::backPropagation(Matrix *dOut, Matrix *input, Layer* previous, double learningRate, double lambdaReg) {

    auto inputT = input->transposed();

    // Current weights derivative with L2 regularization
    auto dW = inputT->multiply(dOut);
    auto dWReg = dW->sum(weights, lambdaReg);

    // update current layer weights
    updateWeights(dWReg, learningRate);

    delete inputT;
    delete dW;
    delete dWReg;

    if(previous == nullptr) {
        return nullptr;
    }

    auto WT = weights->transposed();
    auto dInput = dOut->multiply(WT);

    // gamma and beta derivative for batch norm
    auto inputNorm = previous->getOutputNormalized();
    auto dGammaPartial = dInput->elemMul(inputNorm);
    auto dGamma = dGammaPartial->sumRows();
    auto dBeta = dInput->sumRows();

    // update previous layer batch norm weights
    previous->updateGammaBeta(dGamma, dBeta, learningRate);

    delete dGammaPartial;
    delete dGamma;
    delete dBeta;

    // get input layer final derivative
    auto dInputNorm = getBatchNormDerivative(dInput, previous);

    // get relu derivative
    auto dInputReLU = dInputNorm->ReLUDerivative(input);

    //clear
    delete dInputNorm;
    delete inputNorm;
    delete dInput;
    delete WT;

    return dInputReLU;
}

void Layer::updateWeights(Matrix* dWeights, double learningRate){

    if(frozen)
        return;

    Matrix *newWeights = weights->sum(dWeights, (-1) * learningRate);
    weights->set(newWeights);

    delete newWeights;
}

void Layer::updateGammaBeta(Matrix* dGamma, Matrix* dBeta, double learningRate) {

    if(frozen)
        return;

    assert(dGamma->rows == 1 && dBeta->rows == 1);
    assert ((gamma->columns == dGamma->columns ) && (beta->columns  == dBeta->columns));

    for (int i = 0; i < dGamma->columns; i++) {
        gamma->data[i] -= learningRate * dGamma->data[i];
        beta->data[i] -= learningRate * dBeta->data[i];
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

void Layer::setFrozen(bool value){
    frozen = value;
}
