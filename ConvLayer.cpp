#include "ConvLayer.h"
#include <iostream>

ConvLayer::ConvLayer(int inputChannels, int outputChannels, int stride, int filterSize, int padding, int inputWidth, int inputHeight) {

    this->outputChannels = outputChannels; 
    this->stride = stride;
    this->filterSize = filterSize; 
    this->inputChannels = inputChannels; 
    this->padding = padding;
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
    this->colWidth = ((inputWidth - filterSize + (2 * padding)) / stride) + 1;
    this->colHeight = ((inputHeight - filterSize + (2 * padding)) / stride) + 1;
    this->inputDim = inputWidth * inputHeight * inputChannels;
    this->outputDim = colWidth * colHeight * outputChannels;
    this->colChannels = inputChannels * filterSize * filterSize;;
    this->colInputDim = colChannels * colWidth * colHeight;

    filters = new Matrix(outputChannels, colChannels);
    bias = new Matrix(1, outputChannels);
    filters->randomize();
}

void ConvLayer::BiasAndReLU(Matrix *conv) {
    
    int size = colWidth * colHeight;

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < outputChannels; i++) {
            int index = i * size; 
            for (int j = 0; j < size; j++) {
                conv->data[index] = Matrix::ReLU(conv->data[index] + bias->data[i]);
                index++;
            }
        }
    }
}

Matrix* ConvLayer::feedForward(Matrix *rawInput) {

    delete input;

    input = new Matrix(rawInput->rows, colInputDim);
    auto output = new Matrix(rawInput->rows, outputDim);

    for (int i = 0; i < rawInput->rows; i++) {
        auto colInput = iam2cool(rawInput->data + (i * inputDim), inputChannels, inputWidth, inputHeight, filterSize, stride, padding, colWidth, colHeight, colChannels);
        Matrix::mcopy(input->data + (i * colInputDim), colInput->data, colInputDim);
        //memcpy(input->data + (i * colInputDim), colInput->data, sizeof(float) * colInputDim);
        auto convolution = filters->multiply(colInput);
        BiasAndReLU(convolution);
        //memcpy(output->data + (i * outputDim), convolution->data, outputDim * sizeof(float));
        Matrix::mcopy(output->data + (i * outputDim), convolution->data, outputDim);

        delete colInput;
        delete convolution;
    }

    return output;
}

void ConvLayer::updateWeights(Matrix *dWeights, Matrix *dBias, float learningRate) {
    
    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < outputChannels; i++) {
            int index = i * colChannels;
            for (int i = 0; i < colChannels; i++) {
                filters->data[index] -= dWeights->data[index] * learningRate;
                index++;
            }
            bias->data[i] -= dBias->data[i] * learningRate;
        }

    }
}

Matrix* ConvLayer::backPropagation(Matrix* dOut, float learningRate) {

    assert(dOut->rows == input->rows);

    auto dInputBatch = new Matrix(input->rows, inputDim);

    for (int i = 0; i < input->rows; i++) {
        auto colInput = new Matrix(colChannels, colWidth * colHeight);
        Matrix::mcopy(colInput->data, input->data + (i * colInputDim), colInputDim);
        //memcpy(colInput->data, input->data + (i * colInputDim), sizeof(float) * colInputDim);
        auto colInputT = colInput->transposed();

        auto dOutRow = new Matrix(outputChannels, colWidth * colHeight); // fill

        Matrix::mcopy(dOutRow->data, dOut->data + (i * outputDim), outputDim);
        //memcpy(dOutRow->data, dOut->data + (i * outputDim), outputDim * sizeof(float));

        auto dWeightsRow = dOutRow->multiply(colInputT);
        auto dBiasRow = dOutRow->sumRows();
        
        //update weights
        updateWeights(dWeightsRow, dBiasRow, learningRate);

        delete dWeightsRow;
        delete colInputT;
        delete dBiasRow;

        auto filtersT = filters->transposed();
        auto dInput = filtersT->multiply(dOutRow);
        auto dInputReLU = dInput->ReLUDerivative(colInput);
        auto dInputImage = cool2ami(dInputReLU->data, inputChannels, inputWidth, inputHeight, filterSize, stride, padding, colWidth, colHeight, colChannels);
        dInputBatch->setRow(dInputImage->data, inputDim, i);

        delete colInput;
        delete dInputImage;
        delete filtersT;
        delete dInput;
        delete dInputReLU;
        delete dOutRow;
    }


    return dInputBatch;
}

ConvLayer::~ConvLayer() {
    delete filters;
    delete bias;
    delete input;
}