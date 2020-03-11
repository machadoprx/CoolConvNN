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

    filters = new Matrix(outputChannels, filterSize * filterSize * inputChannels);
    bias = new Matrix(1, outputChannels);
    filters->randomize();
}

void ConvLayer::BiasAndReLU(Matrix *conv, int size) {
    for (int i = 0; i < outputChannels; i++) {
        int index = i * size; 
        for (int j = 0; j < size; j++) {
            conv->data[index] = Matrix::ReLU(conv->data[index] + bias->data[i]);
            index++;
        }
    }
}

#include <iostream>
Matrix* ConvLayer::feedForward(Matrix *rawInput) {

    delete input;
    input = rawInput->copy();

    int colWidth = ((inputWidth - filterSize + (2 * padding)) / stride) + 1;
    int colHeight = ((inputHeight - filterSize + (2 * padding)) / stride) + 1;
    int inputDim = inputWidth * inputHeight * inputChannels;
    int convFilterSize = colWidth * colHeight;
    int convSize = convFilterSize * outputChannels;

    auto output = new Matrix(input->rows, convSize);

    for (int i = 0; i < input->rows; i++) {
        float *img = input->data + (i * inputDim);
        auto colInput = iam2cool(img, inputChannels, inputWidth, inputHeight, filterSize, stride, padding);
        auto convolution = filters->multiply(colInput);
        BiasAndReLU(convolution, convFilterSize);
        memcpy(output->data + (i * convSize), convolution->data, convSize * sizeof(float));

        delete colInput;
        delete convolution;
    }

    return output;
}

void ConvLayer::updateWeights(Matrix *dWeights, Matrix *dBias, float learningRate) {
    
    int len = filterSize * filterSize * inputChannels;

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < outputChannels; i++) {
            int index = i * len;
            for (int i = 0; i < len; i++) {
                filters->data[index] -= dWeights->data[index] * learningRate;
                index++;
            }
            bias->data[i] -= dBias->data[i] * learningRate;
        }

    }
}

Matrix* ConvLayer::backPropagation(Matrix* dOut, float learningRate) {

    assert(dOut->rows == input->rows);

    int colWidth = ((inputWidth - filterSize + (2 * padding)) / stride) + 1;
    int colHeight = ((inputHeight - filterSize + (2 * padding)) / stride) + 1;
    int inputDim = inputWidth * inputHeight * inputChannels;
    int outputDim = colWidth * colHeight * outputChannels;

    auto dInputBatch = new Matrix(input->rows, inputDim);

    for (int i = 0; i < input->rows; i++) {

        float *img = input->data + (i * inputDim);
        auto colInput = iam2cool(img, inputChannels, inputWidth, inputHeight, filterSize, stride, padding);
        auto colInputT = colInput->transposed();

        auto dOutRow = new Matrix(outputChannels, colWidth * colHeight); // fill

        memcpy(dOutRow->data, dOut->data + (i * outputDim), outputDim * sizeof(float));

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
        auto dInputImage = cool2ami(dInputReLU->data, inputChannels, inputWidth, inputHeight, filterSize, stride, padding);
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